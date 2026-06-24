"""
One-time conversion of a Berke Lab sham-feeding session (.pkl) to NWB.

The pickle is a 2-tuple of dicts, one per recording side / bottle:
    element 0 -> Left  / COM3 / mNacSh
    element 1 -> Right / COM4 / NacCore

Each side holds pyPhotometry "3EX_2EM_pulsed" data at 86 Hz plus a large stack of
derived behavioral layers (lick detection / bursts / rates, DLC head-to-spout
distances, engagement states, approach/leave events, hampel QC, ...).

Channel mapping (provided by S. Crater, matches the 3-signal pyPhotometry case in jdb_to_nwb):
    analog_1 -> 470 nm -> gACh4h (green ACh sensor)        signal
    analog_2 -> 565 nm -> rDA3m  (red dopamine sensor)     signal
    analog_3 -> 405 nm -> isosbestic reference (shared)

Style follows https://github.com/calderast/jdb_to_nwb (convert_photometry.py).
"""

import json
import pickle
import uuid
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from pynwb import NWBFile, NWBHDF5IO, TimeSeries
from pynwb.file import Subject
from hdmf.common import DynamicTable

from ndx_fiber_photometry import (
    FiberPhotometryResponseSeries,
    Indicator,
    ExcitationSource,
    OpticalFiber,
    Photodetector,
    DichroicMirror,
    FiberPhotometryTable,
    FiberPhotometry,
)

# ----------------------------------------------------------------------------
# Constants / session-level assumptions  (VERIFY the flagged ones before sharing)
# ----------------------------------------------------------------------------
TZ = ZoneInfo("America/Los_Angeles")
SAMPLING_RATE = 86.0  # Hz, pyPhotometry per-channel rate (from pickle 'sampling_rate')

SPECIES = "Rattus norvegicus"
INSTITUTION = "University of California, San Francisco"
LAB = "Berke Lab"
EXPERIMENTER = ["Slomp, Margo"]

EXPERIMENT_DESCRIPTION = (
    "Sham-feeding sucrose task with dual-region nucleus accumbens fiber photometry. "
    "Two fibers record the green acetylcholine sensor gACh4h (470 nm) and the red "
    "dopamine sensor rDA3m (565 nm) against a 405 nm isosbestic reference, one in medial "
    "NAc shell (mNacSh, left) and one in NAc core (NacCore, right), while licking at a "
    "sucrose spout is detected and the animal's head/nose distance to the spout is tracked."
)
KEYWORDS = ["fiber photometry", "sham feeding", "sucrose", "licking",
            "nucleus accumbens", "dopamine", "acetylcholine", "rDA3m", "gACh4h"]

# analog channel -> (wavelength nm, hampel key, indicator key, role text)
CHANNELS = [
    ("analog_1", 470, "analog1_hampel", "gACh4h",    "gACh4h signal"),
    ("analog_2", 565, "analog2_hampel", "rDA3m",     "rDA3m signal"),
    ("analog_3", 405, "analog3_hampel", "reference", "isosbestic reference"),
]

INDICATOR_INFO = {
    "gACh4h": dict(label="AAV-hSyn-ACh3.8",
                   description="GRAB-ACh3.8 (gACh4h) acetylcholine sensor under the hSyn promoter",
                   manufacturer="BrainVTA"),
    "rDA3m":  dict(label="AAV9-hSyn-rDA3m",
                   description="GRAB rDA3m red-shifted dopamine sensor under the hSyn promoter",
                   manufacturer="BrainVTA"),
}

# Per-side description. idx -> metadata used to disambiguate device/series names.
SIDES = [
    dict(idx=0, side="Left",  com="COM3", region="mNacSh",  hemisphere="left"),
    dict(idx=1, side="Right", com="COM4", region="NacCore", hemisphere="right"),
]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def sanitize(name: str) -> str:
    """Make a string safe/readable for an NWB object name."""
    out = []
    for ch in name:
        if ch.isalnum():
            out.append(ch)
        elif ch == ".":
            out.append("p")
        else:
            out.append("_")
    s = "".join(out)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def to_epoch_seconds(series: pd.Series) -> np.ndarray:
    """Convert a datetime-like column to float epoch seconds (NaT -> NaN)."""
    dt = pd.to_datetime(series, errors="coerce")
    epoch = dt.values.view("int64").astype("float64") / 1e9
    epoch[dt.isna().to_numpy()] = np.nan
    return epoch


def json_str(obj) -> str:
    """JSON-serialize a metadata dict, coercing numpy/datetime to native types."""
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (datetime, pd.Timestamp)):
            return o.isoformat()
        return str(o)
    return json.dumps(obj, default=default)


def pad_to(arr, length):
    """Right-pad a 1D float array with NaN to the requested length."""
    arr = np.asarray(arr, dtype="float64")
    if len(arr) >= length:
        return arr[:length]
    return np.concatenate([arr, np.full(length - len(arr), np.nan)])


# ----------------------------------------------------------------------------
# Build the NWB file
# ----------------------------------------------------------------------------
def build_nwb(pkl_path: Path) -> NWBFile:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    assert isinstance(data, tuple) and len(data) == 2, "Expected a 2-tuple (Left, Right)"

    e0 = data[0]  # reference side for shared session metadata

    # Session start = earliest side start (side 0). Side 1 is offset by a few ms.
    session_start = pd.Timestamp(e0["date_time"]).to_pydatetime().replace(tzinfo=TZ)

    # ---- Subject (from embedded animal metadata) ----
    dob = pd.Timestamp(e0["DOB"]).to_pydatetime().replace(tzinfo=TZ)
    subject = Subject(
        subject_id="IM1923",
        species=SPECIES,
        sex={"Male": "M", "Female": "F"}.get(e0["Sex"], "U"),
        genotype="WT",
        strain=str(e0["Strain"]),
        date_of_birth=dob,
        description=(f"Full animal number {e0['Full_animalNumber']}. "
                     f"Strain {e0['Strain']}. pyPhotometry subject_ID '{e0['subject_ID']}'."),
    )

    session_id = f"IM1923_SF5-Sucrose_{session_start.strftime('%Y%m%d')}"
    notes = (
        f"Sham-feeding sucrose trial SF5. "
        f"Left bottle {e0['Left_COM']} -> {e0['Hit_L']} (target {e0['Target_L']}); "
        f"Right bottle {e0['Right_COM']} -> {data[1]['Hit_R']} (target {data[1]['Target_R']}). "
        f"Grams consumed: {e0['GramConsumed']:.2f} g; grams in pan: {e0['GramInPan']:.2f} g. "
        f"pyPhotometry mode '{e0['mode']}', sampling rate {e0['sampling_rate']} Hz, "
        f"LED current {e0['LED_current']} mA, volts/division {e0['volts_per_division']}."
    )

    nwbfile = NWBFile(
        session_description=("Sham-feeding sucrose task (trial SF5) with dual-region NAc fiber "
                             "photometry (gACh4h 470 nm, rDA3m 565 nm, 405 nm isosbestic) and lick detection."),
        identifier=str(uuid.uuid4()),
        session_start_time=session_start,
        timestamps_reference_time=session_start,
        session_id=session_id,
        institution=INSTITUTION,
        lab=LAB,
        experimenter=EXPERIMENTER,
        experiment_description=EXPERIMENT_DESCRIPTION,
        keywords=KEYWORDS,
        subject=subject,
        notes=notes,
        source_script="convert_sham_feeding_to_nwb.py",
        source_script_file_name="convert_sham_feeding_to_nwb.py",
    )

    # ---- Processing modules ----
    ophys_mod = nwbfile.create_processing_module(
        "ophys", "Filtered and hampel-cleaned photometry signals, plus digital sync.")
    behavior_mod = nwbfile.create_processing_module(
        "behavior", "Lick detection, bottle position, engagement, approach/leave states, lick rates.")
    dlc_mod = nwbfile.create_processing_module(
        "dlc", "DeepLabCut-derived head/nose-to-spout distances and likelihoods.")
    meta_mod = nwbfile.create_processing_module(
        "session_metadata", "Per-side scalar metadata, processing configs and QC parameters as tables.")

    # ---- Photometry devices ----
    exc_sources = {
        470: ExcitationSource(name="Doric Blue LED (470 nm)", illumination_type="LED",
                              excitation_wavelength_in_nm=470.0, manufacturer="Doric", model="ilFMC7-G2"),
        565: ExcitationSource(name="Doric Green LED (565 nm)", illumination_type="LED",
                              excitation_wavelength_in_nm=565.0, manufacturer="Doric", model="ilFMC7-G2"),
        405: ExcitationSource(name="Doric Purple LED (405 nm)", illumination_type="LED",
                              excitation_wavelength_in_nm=405.0, manufacturer="Doric", model="ilFMC7-G2"),
    }
    for src in exc_sources.values():
        nwbfile.add_device(src)

    photodetector = Photodetector(
        name="Doric ilFMC7-G2", detector_type="Silicon photodiode",
        detected_wavelength_in_nm=960.0, manufacturer="Doric", model="ilFMC7-G2",
        description="Integrated LED Fluorescence Mini Cube (5 ports, Gen.2)",
    )
    nwbfile.add_device(photodetector)
    dichroic = DichroicMirror(
        name="Doric ilFMC7-G2 Built-in Dichroic Mirror",
        description="Built-in dichroic mirror of the ilFMC7-G2 minicube", manufacturer="Doric")
    nwbfile.add_device(dichroic)

    fibers, indicators = {}, {}
    for s in SIDES:
        fiber = OpticalFiber(
            name=f"Doric 0.66mm Flat 40mm Optic Fiber ({s['hemisphere']} {s['region']})",
            manufacturer="Doric", model="MFC_200/250-0.66_40mm_MF2.5_FLT",
            numerical_aperture=0.66, core_diameter_in_um=200.0,
            description=f"Recording fiber implanted in {s['hemisphere']} {s['region']}")
        nwbfile.add_device(fiber)
        fibers[s["region"]] = fiber
        for ind_key in ("gACh4h", "rDA3m"):
            info = INDICATOR_INFO[ind_key]
            ind = Indicator(
                name=f"{ind_key} ({s['hemisphere']} {s['region']})",
                label=info["label"], description=info["description"],
                manufacturer=info["manufacturer"], injection_location=s["region"])
            nwbfile.add_device(ind)
            indicators[(s["region"], ind_key)] = ind

    # ---- Fiber photometry table: one row per (side, channel) ----
    fp_table = FiberPhotometryTable(name="fiber_photometry_table",
                                    description="Fiber, indicator and excitation source for each recorded channel.")
    row_index = {}  # (region, analog_key) -> row idx
    next_row = 0
    for s in SIDES:
        region = s["region"]
        for akey, wl, _hk, ind_key, _role in CHANNELS:
            # The 405 isosbestic reference belongs to the gACh4h sensor row-wise.
            indicator_obj = indicators[(region, "gACh4h" if ind_key == "reference" else ind_key)]
            fp_table.add_row(
                location=region,
                optical_fiber=fibers[region],
                photodetector=photodetector,
                dichroic_mirror=dichroic,
                indicator=indicator_obj,
                excitation_source=exc_sources[wl],
            )
            row_index[(region, akey)] = next_row
            next_row += 1
    nwbfile.add_lab_meta_data(FiberPhotometry(name="fiber_photometry", fiber_photometry_table=fp_table))

    def region_of(akey, region):
        return fp_table.create_fiber_photometry_table_region(
            region=[row_index[(region, akey)]], description=f"{akey} @ {region}")

    # ---- Per-side signals & tables ----
    side_meta_rows = []
    for s in SIDES:
        e = data[s["idx"]]
        region = s["region"]
        n = len(e["analog_1"])
        # start offset of this side's stream relative to session_start_time
        side_start = pd.Timestamp(e["date_time"]).to_pydatetime().replace(tzinfo=TZ)
        offset = (side_start - session_start).total_seconds()
        # session-cropped arrays (len 358273) begin at SessionStart_frameNum within the 86 Hz stream
        crop_start = offset + int(e["SessionStart_frameNum"]) / SAMPLING_RATE

        # ----- Raw photometry -> acquisition; filtered + hampel -> ophys module -----
        for akey, wl, hkey, ind_key, role in CHANNELS:
            reg = region_of(akey, region)
            nwbfile.add_acquisition(FiberPhotometryResponseSeries(
                name=f"raw_{wl}_{region}",
                description=f"Raw {role} ({wl} nm) in {region}. pyPhotometry {e['mode']}.",
                data=np.asarray(e[akey], dtype="float64"),
                unit="V", rate=SAMPLING_RATE, starting_time=offset,
                fiber_photometry_table_region=reg))
            ophys_mod.add(FiberPhotometryResponseSeries(
                name=f"filt_{wl}_{region}",
                description=f"pyPhotometry-filtered {role} ({wl} nm) in {region}.",
                data=np.asarray(e[f"{akey}_filt"], dtype="float64"),
                unit="V", rate=SAMPLING_RATE, starting_time=offset,
                fiber_photometry_table_region=region_of(akey, region)))
            ophys_mod.add(FiberPhotometryResponseSeries(
                name=f"hampel_{wl}_{region}",
                description=(f"Hampel-filtered {role} ({wl} nm) in {region} "
                            f"(window {e['QC']['hampel']['window_sec']}s, {e['QC']['hampel']['n_sigmas']} sigma)."),
                data=np.asarray(e[hkey], dtype="float64"),
                unit="V", rate=SAMPLING_RATE, starting_time=offset,
                fiber_photometry_table_region=region_of(akey, region)))

        # ----- Digital sync + rsync pulse times -----
        ophys_mod.add(TimeSeries(
            name=f"digital_sync_{region}", description=f"Digital sync input (rsync) in {region}.",
            data=np.asarray(e["digital_1"], dtype="int8"), unit="n.a.",
            rate=SAMPLING_RATE, starting_time=offset))
        ophys_mod.add(TimeSeries(
            name=f"rsync_pulse_times_{region}",
            description=f"Times of rsync rising edges in {region} (from pyPhotometry pulse_times_1).",
            data=np.ones(len(e["pulse_times_1"]), dtype="int8"), unit="n.a.",
            timestamps=np.asarray(e["pulse_times_1"], dtype="float64") / 1000.0 + offset))

        # ----- Lick / behavior per-sample series (len n, 86 Hz) -----
        full_series = {
            f"lick_binary_{region}": ("LickBinary_2.3", "n.a.",
                "Binary lick detection (vdiff threshold 2.3); NaN outside detection window."),
            f"bottle_position_{region}": ("BottlePos", "n.a.",
                "Bottle present (1) / absent (0); NaN outside session window."),
            f"rsync_from_licks_{region}": ("Rsync_aligned-from-licks", "n.a.",
                "rsync signal aligned from the lick data stream."),
            f"rsync_interp_from_video_{region}": ("rSync_interpolated_from_video", "n.a.",
                "rsync interpolated from the video stream."),
        }
        for name, (key, unit, desc) in full_series.items():
            behavior_mod.add(TimeSeries(name=name, description=desc,
                                        data=np.asarray(e[key], dtype="float64"),
                                        unit=unit, rate=SAMPLING_RATE, starting_time=offset))

        # Engagement state vectors (auto + manual thresholds)
        for key in [k for k in e if k.startswith("Engagement")]:
            behavior_mod.add(TimeSeries(
                name=sanitize(f"{key}_{region}"),
                description=f"Engagement state ('{key}'): animal engaged with spout (1) or not (0), in {region}.",
                data=np.asarray(e[key], dtype="int8"), unit="n.a.",
                rate=SAMPLING_RATE, starting_time=offset))

        # ----- Session-cropped series (len ~358273, begin at SessionStart frame) -----
        cleaned = e["Cleaned_Head_Distance"]
        m = len(cleaned)
        lb = e["LickBurst_Vars_BurstDefinitionILI_basedThresh2000"]
        dse = e["Distance_States_Events"]
        cropped_series = {
            f"cumulative_licks_{region}": (lb["CumLicks"], "n.a.", "Cumulative lick count."),
            f"cleaned_head_distance_{region}": (cleaned, "pixels",
                "Cleaned head-to-spout distance (interpolated, jump-corrected)."),
            f"distance_state_{region}": (dse["state"], "n.a.",
                "Distance state (0/1/2: near/transition/far per QC settings)."),
            f"approach_events_{region}": (dse["Approach_events"], "n.a.", "Approach transition events."),
            f"leave_events_{region}": (dse["Leave_events"], "n.a.", "Leave transition events."),
        }
        for name, (arr, unit, desc) in cropped_series.items():
            behavior_mod.add(TimeSeries(name=name, description=desc,
                                        data=np.asarray(arr), unit=unit,
                                        rate=SAMPLING_RATE, starting_time=crop_start))
        # Labeled burst lick (2D: per-sample burst label)
        behavior_mod.add(TimeSeries(
            name=f"labeled_burst_lick_{region}",
            description="Per-sample burst labeling (column 0: in-burst lick flag, column 1: burst id).",
            data=np.asarray(lb["Labeled_BurstLick"], dtype="float64"), unit="n.a.",
            rate=SAMPLING_RATE, starting_time=crop_start))

        # Derived lick event times (from cumulative-lick increments)
        inc = np.diff(np.asarray(lb["CumLicks"]))
        ev_idx = np.where(inc > 0)[0] + 1
        ev_times = crop_start + ev_idx / SAMPLING_RATE
        behavior_mod.add(TimeSeries(
            name=f"lick_events_{region}",
            description="Detected lick events; data = number of licks registered at each timestamp.",
            data=inc[ev_idx - 1].astype("int16"), unit="licks",
            timestamps=ev_times.astype("float64")))

        # ----- Lick rate time series (1 s / 1 min / 5 min bins) -----
        for nm, key, rate in [("lickrate_1s", "Lickrate_1s", 1.0),
                              ("lickrate_1m", "Lickrate_1m", 1.0 / 60.0),
                              ("lickrate_5m", "Lickrate_5m", 1.0 / 300.0)]:
            behavior_mod.add(TimeSeries(
                name=f"{nm}_{region}",
                description=f"Lick rate in {nm.split('_')[1]} bins (licks/min).",
                data=np.asarray(lb[key], dtype="float64"), unit="licks/min",
                rate=rate, starting_time=crop_start))

        # ----- DLC distances + likelihoods (len n, 86 Hz) -----
        for key in [k for k in e if k.startswith("DLC_")]:
            unit = "pixels" if "Distance" in key else "probability"
            dlc_mod.add(TimeSeries(
                name=sanitize(f"{key}_{region}"),
                description=f"DeepLabCut '{key}' in {region}.",
                data=np.asarray(e[key], dtype="float64"), unit=unit,
                rate=SAMPLING_RATE, starting_time=offset))

        # ----- Per-lick table -----
        n_licks = lb["NumLicks"]
        lick_df = pd.DataFrame({
            "lick_duration_ms": np.asarray(lb["LickDurations_ms"], dtype="float64"),
            "interlick_interval_ms": pad_to(lb["InterlickInterval_ms"], n_licks),
            "ili_startend_ms": pad_to(lb["ILI_startend_ms"], n_licks),
        })
        behavior_mod.add(DynamicTable.from_dataframe(
            df=lick_df, name=f"lick_table_{region}",
            table_description=f"Per-lick durations and inter-lick intervals in {region} ({n_licks} licks)."))

        # ----- Per-burst table -----
        n_bursts = lb["NumBursts"]
        burst_df = pd.DataFrame({
            "full_burst_duration_ms": np.asarray(lb["Full_BurstDur"], dtype="float64"),
            "lick_burst_duration_ms": np.asarray(lb["Lick_BurstDur"], dtype="float64"),
            "avg_licks_per_burst": np.asarray(lb["Avg_LicksPerBurst"], dtype="float64"),
            "ili_between_bursts_ms": pad_to(lb["ILI_betweenBursts"], n_bursts),
            "ili_within_burst_ms_json": [json_str(np.asarray(x).tolist()) for x in lb["ILI_withinBursts"]],
        })
        behavior_mod.add(DynamicTable.from_dataframe(
            df=burst_df, name=f"burst_table_{region}",
            table_description=(f"Per-burst stats in {region} (burst threshold "
                              f"{lb['BurstThreshold_ms']} ms, {n_bursts} bursts).")))

        # ----- Raw lick data table (full video-frame resolution) -----
        raw = e["RawLickData"].copy()
        for col in ("AbsTime", "Abs_time2", "True_Absolute_Time"):
            if col in raw.columns:
                raw[col] = to_epoch_seconds(raw[col])  # epoch seconds (float)
        raw = raw.astype({c: "float64" for c in raw.columns if raw[c].dtype == object}, errors="ignore")
        raw["LickFrames_aligned"] = np.asarray(e["LickFrames_aligned"], dtype="float64")
        behavior_mod.add(DynamicTable.from_dataframe(
            df=raw.reset_index(drop=True), name=f"raw_lick_data_{region}",
            table_description=(f"Raw per-frame lick acquisition in {region}. Datetime columns "
                              "(AbsTime, Abs_time2, True_Absolute_Time) are float epoch seconds.")))

        # ----- Side metadata row -----
        side_meta_rows.append({
            "side": s["side"], "com_port": s["com"], "region": region,
            "hit": e["Hit_L"] if s["idx"] == 0 else e["Hit_R"],
            "target": e["Target_L"] if s["idx"] == 0 else e["Target_R"],
            "full_side_name": e["Full_side_name"],
            "indicator_470nm": "gACh4h", "indicator_565nm": "rDA3m", "reference_405nm": "isosbestic",
            "ppd_filename": e["filename"],
            "mode": e["mode"], "sampling_rate_hz": float(e["sampling_rate"]),
            "led_current_mA_json": json_str(e["LED_current"]),
            "volts_per_division_json": json_str(e["volts_per_division"]),
            "grams_consumed": float(e["GramConsumed"]), "grams_in_pan": float(e["GramInPan"]),
            "num_licks": int(lb["NumLicks"]), "num_bursts": int(lb["NumBursts"]),
            "burst_threshold_ms": int(lb["BurstThreshold_ms"]),
            "session_start_frame": int(e["SessionStart_frameNum"]),
            "session_end_frame": int(e["SessionEnd_frameNum"]),
            "bottle_in_frame": int(e["BottleIn_frameNum"]),
            "n_photometry_samples": int(n), "n_session_samples": int(m),
            "stream_start_offset_s": float(offset),
            "lick_detection_config_json": json_str(e["Processing_params"]["Config_Lickdetection"]),
            "hampel_qc_json": json_str(e["QC"]["hampel"]),
            "distance_states_qc_json": json_str(e["QC"]["distance_states_transition_events"]),
        })

    meta_mod.add(DynamicTable.from_dataframe(
        df=pd.DataFrame(side_meta_rows), name="session_side_metadata",
        table_description="One row per recording side: scalar metadata, processing config and QC parameters (JSON)."))

    return nwbfile


def main():
    pkl_path = Path("IM1923_Trial-SF5-Sucrose_11-06-2025_lickprocessed.pkl")
    out_path = Path(f"IM1923_SF5-Sucrose_20251106.nwb")
    print(f"Reading {pkl_path} ...")
    nwbfile = build_nwb(pkl_path)
    print(f"Writing {out_path} ...")
    with NWBHDF5IO(out_path, mode="w") as io:
        io.write(nwbfile)
    print("Done.")

    # Read back as a sanity check
    with NWBHDF5IO(out_path, mode="r") as io:
        nwb = io.read()
        print("Re-read OK:", nwb.session_id)
        print("  acquisition:", len(nwb.acquisition), "objects")
        print("  ophys:", len(nwb.processing["ophys"].data_interfaces))
        print("  behavior:", len(nwb.processing["behavior"].data_interfaces))
        print("  dlc:", len(nwb.processing["dlc"].data_interfaces))


if __name__ == "__main__":
    main()
