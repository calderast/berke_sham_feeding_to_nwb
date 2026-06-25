"""
One-time conversion of a Berke Lab sham-feeding session (.pkl) to NWB.

The pickle is a 2-tuple of dicts, one per recording side / bottle. Each dict's
side/COM/region identity is read from its own 'Full_side_name' field (e.g.
'COM3_Left_mNacSh'), NOT assumed from tuple position -- the region<->side mapping
differs between sessions (e.g. IM1923 has Left=mNacSh/Right=NacCore, while IM1929
has them swapped, Left=NacCore/Right=mNacSh).

Each side holds pyPhotometry "3EX_2EM_pulsed" data at 86 Hz plus a large stack of
derived behavioral layers (lick detection / bursts / rates, DLC head-to-spout
distances, engagement states, approach/leave events, hampel QC, ...).

Channel mapping (provided by S. Crater, matches the 3-signal pyPhotometry case in jdb_to_nwb):
    analog_1 -> 470 nm -> gACh4h (green ACh sensor)        signal
    analog_2 -> 565 nm -> rDA3m  (red dopamine sensor)     signal
    analog_3 -> 405 nm -> gACh4h reference

Style follows https://github.com/calderast/jdb_to_nwb (convert_photometry.py).
"""

import json
import pickle
import re
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
    "Two fibers record the green acetylcholine sensor gACh4h (470 nm, with its own 405 nm "
    "reference channel) and the red dopamine sensor rDA3m (565 nm, no reference channel), one in "
    "NAc core (NacCore) and one in medial NAc shell (mNacSh), while licking at a sucrose spout is "
    "detected and the animal's head/nose distance to the spout is tracked. "
    "Task structure: 10 min baseline (no bottle available), then 60 x 30 s spout-access periods each "
    "followed by 30 s of no access; a 350 ms 4000 Hz tone marks each trial start. The animal is sham-fed "
    "0.8 M sucrose (sham feeding is considered successful when liquid collected in the pan is >= 50% of "
    "the liquid consumed). Apparatus: MedAssociates retractable sipper with Lixit + bottle (ENV-252M) and "
    "MedAssociates grid-floor harness; controlled by Bonsai (incl. video recording), Arduino (lick "
    "detection and moving spout), and Python with custom scripts. Animals were flushed with heated "
    "(~body temp) 0.9% NaCl."
)
KEYWORDS = ["fiber photometry", "sham feeding", "sucrose", "licking",
            "nucleus accumbens", "dopamine", "acetylcholine", "rDA3m", "gACh4h"]

# Virus / surgery descriptions (same construct + targets for both animals; coordinates are for males).
VIRUS = (
    "1:1 mix of two GRAB sensors, each injected undiluted from stock: "
    "AAV-hSyn-ACh4h3.8 (gACh4h acetylcholine sensor, 1.15e13 vg/mL, BrainVTA) and "
    "AAV9-hSyn-rDA3m (red-shifted dopamine sensor, 5.89e12 vg/mL, BrainVTA)."
)
SURGERY = (
    "Bilateral NAc fiber photometry. Target coordinates (male, mm from bregma; ML +/- for right/left): "
    "NAc core AP +1.7, ML +/-1.7, DV 6.8 from dura (fiber) / 7.0 (virus), no angle; "
    "medial NAc shell AP +1.3, ML +/-1.6, DV 6.2 from dura (fiber) / 6.4 (virus), 6 degree angle. "
    "Doric 200 um fibers (B280-2615-10, MFC_200/250-0.66_10mm_MF2.5_FLT)."
)

# analog channel -> (wavelength nm, hampel key, indicator key, role text)
CHANNELS = [
    ("analog_1", 470, "analog1_hampel", "gACh4h",    "gACh4h signal"),
    ("analog_2", 565, "analog2_hampel", "rDA3m",     "rDA3m signal"),
    ("analog_3", 405, "analog3_hampel", "reference", "gACh4h reference"),
]

INDICATOR_INFO = {
    "gACh4h": dict(label="AAV-hSyn-ACh4h3.8",
                   description=("GRAB gACh4h3.8 acetylcholine sensor under the hSyn promoter. "
                                "Titer 1.15e13 vg/mL (BrainVTA). Injected undiluted from stock, "
                                "in a 1:1 mix with rDA3m."),
                   manufacturer="BrainVTA"),
    "rDA3m":  dict(label="AAV9-hSyn-rDA3m",
                   description=("GRAB rDA3m red-shifted dopamine sensor under the hSyn promoter. "
                                "Titer 5.89e12 vg/mL (BrainVTA). Injected undiluted from stock, "
                                "in a 1:1 mix with gACh4h."),
                   manufacturer="BrainVTA"),
}

# Stereotaxic targets (male), mm from bregma. ML magnitude; sign is applied per hemisphere.
# DV differs for the fiber tip (recording) vs the virus injection.
COORDS = {
    "NacCore": dict(ap=1.7, ml=1.7, dv_fiber=6.8, dv_virus=7.0, angle_deg=0),
    "mNacSh":  dict(ap=1.3, ml=1.6, dv_fiber=6.2, dv_virus=6.4, angle_deg=6),
}



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


def parse_side(idx, e):
    """Derive a side's identity from the pickle dict itself (no hard-coding).

    'Full_side_name' is formatted 'COM3_Left_mNacSh' / 'COM4_Right_NacCore'.
    Hit_*/Target_* and the COM ports are read from the matching L/R fields.
    """
    parts = str(e["Full_side_name"]).split("_")
    com, side, region = parts[0], parts[1], "_".join(parts[2:])
    hemisphere = "left" if side.lower().startswith("l") else "right"
    hit = e["Hit_L"] if side == "Left" else e["Hit_R"]
    target = e["Target_L"] if side == "Left" else e["Target_R"]
    return dict(idx=idx, side=side, com=com, region=region,
                hemisphere=hemisphere, hit=hit, target=target)


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

    # Read identity from the pickle rather than hard-coding it.
    sides = [parse_side(i, d) for i, d in enumerate(data)]
    animal_name = str(e0["subject_ID"]).split("_")[0]                  # e.g. "IM1923"
    mt = re.search(r"Trial[-_](.+?)_COM", str(e0["filename"]))          # e.g. "SF5-Sucrose"
    trial_label = mt.group(1) if mt else "session"

    # ---- Subject (from embedded animal metadata) ----
    dob = pd.Timestamp(e0["DOB"]).to_pydatetime().replace(tzinfo=TZ)
    subject = Subject(
        subject_id=animal_name,
        species=SPECIES,
        sex={"Male": "M", "Female": "F"}.get(e0["Sex"], "U"),
        genotype=str(e0["Strain"]),
        strain=str(e0["Strain"]),
        date_of_birth=dob,
        description=(f"Full animal number {e0['Full_animalNumber']}. "
                     f"Strain {e0['Strain']}. pyPhotometry subject_ID '{e0['subject_ID']}'."),
    )

    session_id = f"{animal_name}_{trial_label}_{session_start.strftime('%Y%m%d')}"
    side_desc = "; ".join(
        f"{sd['side']} bottle {sd['com']} -> {sd['region']} (target {sd['target']})" for sd in sides)
    notes = (
        f"Sham-feeding {trial_label} trial. {side_desc}. "
        f"Grams consumed: {e0['GramConsumed']:.2f} g; grams in pan: {e0['GramInPan']:.2f} g. "
        f"pyPhotometry mode '{e0['mode']}', sampling rate {e0['sampling_rate']} Hz, "
        f"LED current {e0['LED_current']} mA, volts/division {e0['volts_per_division']}."
    )

    nwbfile = NWBFile(
        session_description=(f"Sham-feeding {trial_label} task with dual-region NAc fiber "
                             "photometry (gACh4h 470 nm, rDA3m 565 nm, 405 nm gACh4h reference) and lick detection."),
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
        surgery=SURGERY,
        virus=VIRUS,
        source_script="convert_sham_feeding_to_nwb.py",
        source_script_file_name="convert_sham_feeding_to_nwb.py",
    )

    # ---- Processing modules ----
    behavior_mod = nwbfile.create_processing_module(
        "behavior", "Lick detection, bottle position, engagement, approach/leave states, lick rates.")
    dlc_mod = nwbfile.create_processing_module(
        "dlc", "DeepLabCut-derived head/nose-to-spout distances and likelihoods.")
    meta_mod = nwbfile.create_processing_module(
        "session_metadata", "Per-side scalar metadata, processing configs and QC parameters as tables.")

    # ---- Photometry devices (Thorlabs fiber-coupled LEDs -> Doric FMC6 minicube -> Doric detector) ----
    exc_sources = {
        470: ExcitationSource(name="Thorlabs 470 nm LED", illumination_type="LED",
                              excitation_wavelength_in_nm=470.0, manufacturer="Thorlabs", model="M470F3",
                              description="470 nm fiber-coupled LED, 17.2 mW (min), 1000 mA, SMA"),
        565: ExcitationSource(name="Thorlabs 565 nm LED", illumination_type="LED",
                              excitation_wavelength_in_nm=565.0, manufacturer="Thorlabs", model="M565F3",
                              description="565 nm fiber-coupled LED, 9.9 mW (min), 700 mA, SMA"),
        405: ExcitationSource(name="Thorlabs 405 nm LED", illumination_type="LED",
                              excitation_wavelength_in_nm=405.0, manufacturer="Thorlabs", model="M405F3",
                              description=("405 nm fiber-coupled LED, 3.0 mW (min), 500 mA, SMA. Driven by a "
                                           "separate Thorlabs LED driver (PyBoard controls only 2 LEDs).")),
    }
    for src in exc_sources.values():
        nwbfile.add_device(src)

    photodetector = Photodetector(
        name="Doric Fluorescence Detector", detector_type="Silicon photodiode",
        detected_wavelength_in_nm=600.0, manufacturer="Doric", serial_number="192201-01",
        description="Doric fluorescence detector (s/n 192201-01).",
    )
    nwbfile.add_device(photodetector)
    # The Doric FMC6 minicube provides the excitation/emission/dichroic filtering; represented as a DichroicMirror.
    dichroic = DichroicMirror(
        name="Doric FMC6 Minicube",
        manufacturer="Doric",
        model=("FMC6_IE(400-410)_E1(460-490)_F1(500-540)_E2(555-570)_F2(580-680)_S"),
        description=("Doric 6-port Fluorescence Mini Cube (GCaMP + red fluorophore), Gen 1 (~2015). "
                     "Filter bands (nm): isosbestic exc 400-410, exc1 460-490, em1 500-540, "
                     "exc2 555-570, em2 580-680. FC connectors on all ports."))
    nwbfile.add_device(dichroic)

    fibers, indicators = {}, {}
    for s in sides:
        region, hemi = s["region"], s["hemisphere"]
        c = COORDS[region]
        ml = (-c["ml"] if hemi == "left" else c["ml"])
        fiber_coords = (c["ap"], ml, c["dv_fiber"])     # AP, ML, DV (mm) of the fiber tip
        virus_coords = (c["ap"], ml, c["dv_virus"])     # AP, ML, DV (mm) of the virus injection
        s["fiber_coords"] = fiber_coords

        angle_txt = "no angle" if c["angle_deg"] == 0 else f"{c['angle_deg']} degree angle"
        fiber = OpticalFiber(
            name=f"Doric 200um 10mm Optic Fiber ({hemi} {region})",
            manufacturer="Doric", model="MFC_200/250-0.66_10mm_MF2.5_FLT",
            numerical_aperture=0.66, core_diameter_in_um=200.0,
            description=(f"Doric 200 um fiber (B280-2615-10) implanted in {hemi} {region} at "
                         f"AP {fiber_coords[0]}, ML {fiber_coords[1]}, DV {fiber_coords[2]} mm from dura ({angle_txt})."))
        nwbfile.add_device(fiber)
        fibers[region] = fiber
        for ind_key in ("gACh4h", "rDA3m"):
            info = INDICATOR_INFO[ind_key]
            ind = Indicator(
                name=f"{ind_key} ({hemi} {region})",
                label=info["label"], description=info["description"],
                manufacturer=info["manufacturer"], injection_location=region,
                injection_coordinates_in_mm=virus_coords)
            nwbfile.add_device(ind)
            indicators[(region, ind_key)] = ind

    # ---- Fiber photometry table: one row per (side, channel) ----
    fp_table = FiberPhotometryTable(name="fiber_photometry_table",
                                    description="Fiber, indicator and excitation source for each recorded channel.")
    row_index = {}  # (region, analog_key) -> row idx
    next_row = 0
    for s in sides:
        region = s["region"]
        for akey, wl, _hk, ind_key, _role in CHANNELS:
            # The 405 reference channel belongs to gACh4h only (not rDA3m, which has no reference).
            indicator_obj = indicators[(region, "gACh4h" if ind_key == "reference" else ind_key)]
            fp_table.add_row(
                location=region,
                coordinates=s["fiber_coords"],  # AP, ML, DV (mm) of the recording fiber tip
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
    for s in sides:
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
            nwbfile.add_acquisition(FiberPhotometryResponseSeries(
                name=f"filt_{wl}_{region}",
                description=f"pyPhotometry-filtered {role} ({wl} nm) in {region}.",
                data=np.asarray(e[f"{akey}_filt"], dtype="float64"),
                unit="V", rate=SAMPLING_RATE, starting_time=offset,
                fiber_photometry_table_region=region_of(akey, region)))
            nwbfile.add_acquisition(FiberPhotometryResponseSeries(
                name=f"hampel_{wl}_{region}",
                description=(f"Hampel-filtered {role} ({wl} nm) in {region} "
                            f"(window {e['QC']['hampel']['window_sec']}s, {e['QC']['hampel']['n_sigmas']} sigma)."),
                data=np.asarray(e[hkey], dtype="float64"),
                unit="V", rate=SAMPLING_RATE, starting_time=offset,
                fiber_photometry_table_region=region_of(akey, region)))

        # ----- Digital sync + rsync pulse times -----
        nwbfile.add_acquisition(TimeSeries(
            name=f"digital_sync_{region}", description=f"Digital sync input (rsync) in {region}.",
            data=np.asarray(e["digital_1"], dtype="int8"), unit="n.a.",
            rate=SAMPLING_RATE, starting_time=offset))
        nwbfile.add_acquisition(TimeSeries(
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
            "side": s["side"], "com_port": s["com"], "region": region, "hemisphere": s["hemisphere"],
            "hit": s["hit"], "target": s["target"],
            "full_side_name": e["Full_side_name"],
            "indicator_470nm": "gACh4h", "indicator_565nm": "rDA3m", "reference_405nm": "gACh4h reference",
            "fiber_coords_ap_ml_dv_mm_json": json_str(list(s["fiber_coords"])),
            "virus_coords_ap_ml_dv_mm_json": json_str([COORDS[region]["ap"],
                                                       s["fiber_coords"][1], COORDS[region]["dv_virus"]]),
            "implant_angle_deg": int(COORDS[region]["angle_deg"]),
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


def convert_one(pkl_path: Path):
    print(f"Reading {pkl_path} ...")
    nwbfile = build_nwb(pkl_path)
    out_path = pkl_path.parent / f"{nwbfile.session_id}.nwb"  # data-driven name from session_id
    print(f"Writing {out_path} ...")
    with NWBHDF5IO(out_path, mode="w") as io:
        io.write(nwbfile)

    # Read back as a sanity check
    with NWBHDF5IO(out_path, mode="r") as io:
        nwb = io.read()
        print(f"Done. Re-read OK: {nwb.session_id} | "
              f"acquisition: {len(nwb.acquisition)}, "
              f"behavior: {len(nwb.processing['behavior'].data_interfaces)}, "
              f"dlc: {len(nwb.processing['dlc'].data_interfaces)}")


def main():
    import sys
    # Convert the pickle(s) given on the command line, or every *_lickprocessed.pkl
    # in this directory if none are given.
    args = [Path(p) for p in sys.argv[1:]]
    pkl_paths = args or sorted(Path(__file__).parent.glob("*_lickprocessed.pkl"))
    if not pkl_paths:
        print("No pickle files given and no *_lickprocessed.pkl found in this directory.")
        return
    for pkl_path in pkl_paths:
        convert_one(pkl_path)


if __name__ == "__main__":
    main()
