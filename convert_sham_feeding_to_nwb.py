"""
One-time conversion of a Berke Lab sham-feeding session (.pkl) to NWB.

The pickle is a 2-tuple of dicts, one per recording hemisphere (one optical fiber per hemisphere).
Each dict's side/COM/region identity is read from its own 'Full_side_name' field (e.g. 'COM3_Left_mNacSh').
The region<->side mapping differs between sessions (e.g. IM1923 has Left=mNacSh/Right=NacCore, 
while IM1929 has Left=NacCore/Right=mNacSh).

Each side holds pyPhotometry "3EX_2EM_pulsed" data at 86 Hz plus a large stack of
derived behavioral layers (lick detection / bursts / rates, DLC head-to-spout
distances, engagement states, approach/leave events, hampel QC, ...).

PyPhotometry channel mapping:
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


## Constants

TZ = ZoneInfo("America/Los_Angeles")
SAMPLING_RATE = 86.0  # Hz, pyPhotometry per-channel rate

SPECIES = "Rattus norvegicus"
INSTITUTION = "University of California, San Francisco"
LAB = "Berke Lab"
EXPERIMENTER = ["Slomp, Margo"]

EXPERIMENT_DESCRIPTION = (
    "Sham-feeding sucrose task with dual-region nucleus accumbens fiber photometry. "
    "Two fibers record the green acetylcholine sensor gACh4h (470 nm, with 405 nm "
    "reference channel) and the red dopamine sensor rDA3m (565 nm, no reference channel) from two "
    "nucleus accumbens sites (core and/or medial shell; see the 'surgery' field for this session's "
    "targets), while licking at a sucrose spout is detected and the animal's head/nose distance to "
    "the spout is tracked. "
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

# Virus info (same constructs and coords for all animals)
VIRUS = (
    "1:1 mix of two GRAB sensors, each injected undiluted from stock: "
    "AAV-hSyn-ACh4h3.8 (gACh4h acetylcholine sensor, 1.15e13 vg/mL, BrainVTA) and "
    "AAV9-hSyn-rDA3m (red-shifted dopamine sensor, 5.89e12 vg/mL, BrainVTA)."
)

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

# Stereotaxic coordinates (male rats only), in mm from bregma. 
# ML coordinate here is magnitude only (sign is applied per hemisphere)
# DV differs for the fiber tip (recording site) vs the virus injection site.
COORDS = {
    "NacCore": dict(ap=1.7, ml=1.7, dv_fiber=6.8, dv_virus=7.0, angle_deg=0),
    "mNacSh":  dict(ap=1.3, ml=1.6, dv_fiber=6.2, dv_virus=6.4, angle_deg=6),
}

# Readable region names for the surgery description
REGION_DISPLAY = {"NacCore": "NAc core", "mNacSh": "medial NAc shell"}

# per pyPhotometry channel: (pickle key, excitation wavelength nm, hampel-cleaned key, indicator key, description)
CHANNELS = [
    ("analog_1", 470, "analog1_hampel", "gACh4h",    "gACh4h signal"),
    ("analog_2", 565, "analog2_hampel", "rDA3m",     "rDA3m signal"),
    ("analog_3", 405, "analog3_hampel", "reference", "gACh4h reference"),
]


## Helpers

def sanitize(name: str) -> str:
    """Make a string safe/readable for an NWB object name."""
    cleaned_chars = []
    for char in name:
        if char.isalnum():
            # alphanumeric ok
            cleaned_chars.append(char)
        elif char == ".":
            # periods aren't allowed in nwb names so use a "p" instead
            cleaned_chars.append("p")
        else:
            # everything else (/, :, etc) gets replaced with an underscore
            cleaned_chars.append("_")
    cleaned_name = "".join(cleaned_chars)
    # remove duplicate underscores
    while "__" in cleaned_name:
        cleaned_name = cleaned_name.replace("__", "_")
    return cleaned_name.strip("_")


def to_epoch_seconds(datetime_series: pd.Series) -> np.ndarray:
    """Convert a datetime-like column to float epoch seconds (NaT -> NaN)."""
    datetimes = pd.to_datetime(datetime_series, errors="coerce")
    epoch_seconds = datetimes.values.view("int64").astype("float64") / 1e9
    epoch_seconds[datetimes.isna().to_numpy()] = np.nan
    return epoch_seconds


def json_str(obj) -> str:
    """JSON-serialize a metadata object, coercing numpy/datetime to native types."""
    def coerce(value):
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (datetime, pd.Timestamp)):
            return value.isoformat()
        return str(value)
    return json.dumps(obj, default=coerce)


def pad_to(array, length):
    """Right-pad a 1D float array with NaN to the requested length."""
    array = np.asarray(array, dtype="float64")
    if len(array) >= length:
        return array[:length]
    return np.concatenate([array, np.full(length - len(array), np.nan)])


def parse_side(index, side_data):
    """Derive a recording side's identity from the pickle dict.

    Here 'side' is the recording hemisphere (one optical fiber per hemisphere) and
    which port it was recorded from (COM3 or COM4).

    'Full_side_name' is formatted 'COM3_Left_mNacSh' / 'COM4_Right_NacCore'.
    Hit_*/Target_* and the COM ports are read from the matching L/R fields.
    """
    name_parts = str(side_data["Full_side_name"]).split("_")
    com_port, side_name, region = name_parts[0], name_parts[1], "_".join(name_parts[2:])
    hemisphere = "left" if side_name.lower().startswith("l") else "right"
    hit = side_data["Hit_L"] if side_name == "Left" else side_data["Hit_R"]
    target = side_data["Target_L"] if side_name == "Left" else side_data["Target_R"]
    return dict(index=index, side=side_name, com=com_port, region=region,
                hemisphere=hemisphere, hit=hit, target=target)


def build_surgery(sides):
    """Build the surgery description for the nwb based on the regions targeted this session.

    Only the regions present in `sides` are described, with coordinates for core/shell pulled from COORDS.
    Reports the ML sign(s) for the hemisphere(s) actually implanted (negative for left hemisphere).
    """
    hemispheres_by_region = {}  # region -> set of hemispheres, preserving first-seen order
    for side in sides:
        hemispheres_by_region.setdefault(side["region"], set()).add(side["hemisphere"])

    target_descriptions = []
    for region, hemispheres in hemispheres_by_region.items():
        coords = COORDS[region]
        region_name = REGION_DISPLAY.get(region, region)
        if hemispheres == {"left", "right"}:
            hemisphere_label, ml_label = "bilateral", f"ML +/-{coords['ml']}"
        elif hemispheres == {"left"}:
            hemisphere_label, ml_label = "left", f"ML -{coords['ml']}"
        else:
            hemisphere_label, ml_label = "right", f"ML +{coords['ml']}"
        angle_label = "no angle" if coords["angle_deg"] == 0 else f"{coords['angle_deg']} degree angle"
        target_descriptions.append(
            f"{hemisphere_label} {region_name} AP +{coords['ap']}, {ml_label}, "
            f"DV {coords['dv_fiber']} from dura (fiber) / {coords['dv_virus']} (virus), {angle_label}")

    return (f"NAc fiber photometry. Target coordinates (male, mm from bregma): "
            f"{'; '.join(target_descriptions)}. "
            "Doric 200 um fibers (B280-2615-10, MFC_200/250-0.66_10mm_MF2.5_FLT).")


## Build the NWB file

def build_nwb(pkl_path: Path) -> NWBFile:
    with open(pkl_path, "rb") as pkl_file:
        sides_data = pickle.load(pkl_file)
    assert isinstance(sides_data, tuple) and len(sides_data) == 2, "Expected a 2-tuple (Left, Right)"

    # Use the first side for shared session-level metadata (subject info, grams consumed, etc)
    first_side_data = sides_data[0]

    # Session start = earliest side start (side 0). The other side is offset by a few ms.
    session_start = pd.Timestamp(first_side_data["date_time"]).to_pydatetime().replace(tzinfo=TZ)

    # Get sides from the pickle
    sides = [parse_side(index, side_data) for index, side_data in enumerate(sides_data)]
    animal_name = str(first_side_data["subject_ID"]).split("_")[0]                      # e.g. "IM1923"
    trial_match = re.search(r"Trial[-_](.+?)_COM", str(first_side_data["filename"]))    # e.g. "SF5-Sucrose"
    trial_label = trial_match.group(1) if trial_match else "session"

    # Get subject info from pickle
    date_of_birth = pd.Timestamp(first_side_data["DOB"]).to_pydatetime().replace(tzinfo=TZ)
    subject = Subject(
        subject_id=animal_name,
        species=SPECIES,
        sex={"Male": "M", "Female": "F"}.get(first_side_data["Sex"], "U"),
        genotype=str(first_side_data["Strain"]),
        strain=str(first_side_data["Strain"]),
        date_of_birth=date_of_birth,
        description=(f"Full animal number {first_side_data['Full_animalNumber']}. "
                     f"Strain {first_side_data['Strain']}. "
                     f"pyPhotometry subject_ID '{first_side_data['subject_ID']}'."),
    )

    session_id = f"{animal_name}_{trial_label}_{session_start.strftime('%Y%m%d')}"
    hemisphere_descriptions = "; ".join(
        f"{side['side']} hemisphere ({side['com']}) recorded from {side['region']} (target {side['target']})"
        for side in sides)
    notes = (
        f"Sham-feeding {trial_label} trial. {hemisphere_descriptions}. "
        f"Grams consumed: {first_side_data['GramConsumed']:.2f} g; "
        f"grams in pan: {first_side_data['GramInPan']:.2f} g. "
        f"pyPhotometry mode '{first_side_data['mode']}', sampling rate {first_side_data['sampling_rate']} Hz, "
        f"LED current {first_side_data['LED_current']} mA, "
        f"volts/division {first_side_data['volts_per_division']}."
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
        surgery=build_surgery(sides),
        virus=VIRUS,
        source_script="convert_sham_feeding_to_nwb.py",
        source_script_file_name="convert_sham_feeding_to_nwb.py",
    )

    # Processing modules
    behavior_module = nwbfile.create_processing_module(
        "behavior", "Lick detection, bottle position, engagement, approach/leave states, lick rates.")
    dlc_module = nwbfile.create_processing_module(
        "dlc", "DeepLabCut-derived head/nose-to-spout distances and likelihoods.")
    metadata_module = nwbfile.create_processing_module(
        "session_metadata", "Per-side scalar metadata, processing configs and QC parameters as tables.")

    # Photometry devices (Thorlabs fiber-coupled LEDs -> Doric FMC6 minicube -> Doric detector)
    excitation_sources = {
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
    for excitation_source in excitation_sources.values():
        nwbfile.add_device(excitation_source)

    photodetector = Photodetector(
        name="Doric Fluorescence Detector", detector_type="Silicon photodiode",
        detected_wavelength_in_nm=600.0, manufacturer="Doric", serial_number="192201-01",
        description="Doric fluorescence detector (s/n 192201-01).",
    )
    nwbfile.add_device(photodetector)
    # The Doric FMC6 minicube provides the excitation/emission/dichroic filtering; represented as a DichroicMirror.
    dichroic_mirror = DichroicMirror(
        name="Doric FMC6 Minicube",
        manufacturer="Doric",
        model="FMC6_IE(400-410)_E1(460-490)_F1(500-540)_E2(555-570)_F2(580-680)_S",
        description=("Doric 6-port Fluorescence Mini Cube (GCaMP + red fluorophore), Gen 1 (~2015). "
                     "Filter bands (nm): isosbestic exc 400-410, exc1 460-490, em1 500-540, "
                     "exc2 555-570, em2 580-680. FC connectors on all ports."))
    nwbfile.add_device(dichroic_mirror)

    # One optical fiber per region, and two indicators (gACh4h, rDA3m) injected at each fiber site.
    fibers_by_region = {}
    indicators_by_region_and_sensor = {}
    for side in sides:
        region, hemisphere = side["region"], side["hemisphere"]
        coords = COORDS[region]
        ml_mm = -coords["ml"] if hemisphere == "left" else coords["ml"]
        fiber_coords = (coords["ap"], ml_mm, coords["dv_fiber"])  # AP, ML, DV (mm) of the fiber tip
        virus_coords = (coords["ap"], ml_mm, coords["dv_virus"])  # AP, ML, DV (mm) of the virus injection
        side["fiber_coords"] = fiber_coords

        angle_label = "no angle" if coords["angle_deg"] == 0 else f"{coords['angle_deg']} degree angle"
        fiber = OpticalFiber(
            name=f"Doric 200um 10mm Optic Fiber ({hemisphere} {region})",
            manufacturer="Doric", model="MFC_200/250-0.66_10mm_MF2.5_FLT",
            numerical_aperture=0.66, core_diameter_in_um=200.0,
            description=(f"Doric 200 um fiber (B280-2615-10) implanted in {hemisphere} {region} at "
                         f"AP {fiber_coords[0]}, ML {fiber_coords[1]}, DV {fiber_coords[2]} mm "
                         f"from dura ({angle_label})."))
        nwbfile.add_device(fiber)
        fibers_by_region[region] = fiber

        for sensor in ("gACh4h", "rDA3m"):
            indicator_info = INDICATOR_INFO[sensor]
            indicator = Indicator(
                name=f"{sensor} ({hemisphere} {region})",
                label=indicator_info["label"], description=indicator_info["description"],
                manufacturer=indicator_info["manufacturer"], injection_location=region,
                injection_coordinates_in_mm=virus_coords)
            nwbfile.add_device(indicator)
            indicators_by_region_and_sensor[(region, sensor)] = indicator

    # Fiber photometry table: one row per (side, channel)
    fiber_table = FiberPhotometryTable(
        name="fiber_photometry_table",
        description="Fiber, indicator and excitation source for each recorded channel.")
    row_index_by_channel = {}  # (region, analog_key) -> row index
    next_row_index = 0
    for side in sides:
        region = side["region"]
        for analog_key, wavelength, _hampel_key, indicator_key, _role in CHANNELS:
            # The 405 reference channel belongs to gACh4h only (not rDA3m, which has no reference).
            sensor = "gACh4h" if indicator_key == "reference" else indicator_key
            fiber_table.add_row(
                location=region,
                coordinates=side["fiber_coords"],  # AP, ML, DV (mm) of the recording fiber tip
                optical_fiber=fibers_by_region[region],
                photodetector=photodetector,
                dichroic_mirror=dichroic_mirror,
                indicator=indicators_by_region_and_sensor[(region, sensor)],
                excitation_source=excitation_sources[wavelength],
            )
            row_index_by_channel[(region, analog_key)] = next_row_index
            next_row_index += 1
    nwbfile.add_lab_meta_data(FiberPhotometry(name="fiber_photometry", fiber_photometry_table=fiber_table))

    def channel_table_region(analog_key, region):
        """A single-row FiberPhotometryTableRegion pointing at this channel's table row."""
        return fiber_table.create_fiber_photometry_table_region(
            region=[row_index_by_channel[(region, analog_key)]], description=f"{analog_key} @ {region}")

    # Per-side signals & tables
    side_metadata_rows = []
    for side in sides:
        side_data = sides_data[side["index"]]
        region = side["region"]
        n_samples = len(side_data["analog_1"])

        # Offset of this side's photometry stream relative to session_start_time (seconds).
        side_start_time = pd.Timestamp(side_data["date_time"]).to_pydatetime().replace(tzinfo=TZ)
        stream_offset_s = (side_start_time - session_start).total_seconds()
        # Session-cropped arrays (len ~358273) begin at SessionStart_frameNum within the 86 Hz stream.
        session_crop_start_s = stream_offset_s + int(side_data["SessionStart_frameNum"]) / SAMPLING_RATE

        # Photometry: raw, pyPhotometry-filtered, and hampel-cleaned -- all to acquisition.
        for analog_key, wavelength, hampel_key, _indicator_key, role in CHANNELS:
            nwbfile.add_acquisition(FiberPhotometryResponseSeries(
                name=f"raw_{wavelength}_{region}",
                description=f"Raw {role} ({wavelength} nm) in {region}. pyPhotometry {side_data['mode']}.",
                data=np.asarray(side_data[analog_key], dtype="float64"),
                unit="V", rate=SAMPLING_RATE, starting_time=stream_offset_s,
                fiber_photometry_table_region=channel_table_region(analog_key, region)))
            nwbfile.add_acquisition(FiberPhotometryResponseSeries(
                name=f"filt_{wavelength}_{region}",
                description=f"pyPhotometry-filtered {role} ({wavelength} nm) in {region}.",
                data=np.asarray(side_data[f"{analog_key}_filt"], dtype="float64"),
                unit="V", rate=SAMPLING_RATE, starting_time=stream_offset_s,
                fiber_photometry_table_region=channel_table_region(analog_key, region)))
            nwbfile.add_acquisition(FiberPhotometryResponseSeries(
                name=f"hampel_{wavelength}_{region}",
                description=(f"Hampel-filtered {role} ({wavelength} nm) in {region} "
                            f"(window {side_data['QC']['hampel']['window_sec']}s, "
                            f"{side_data['QC']['hampel']['n_sigmas']} sigma)."),
                data=np.asarray(side_data[hampel_key], dtype="float64"),
                unit="V", rate=SAMPLING_RATE, starting_time=stream_offset_s,
                fiber_photometry_table_region=channel_table_region(analog_key, region)))

        # Digital sync + rsync pulse times
        nwbfile.add_acquisition(TimeSeries(
            name=f"digital_sync_{region}", description=f"Digital sync input (rsync) in {region}.",
            data=np.asarray(side_data["digital_1"], dtype="int8"), unit="n.a.",
            rate=SAMPLING_RATE, starting_time=stream_offset_s))
        nwbfile.add_acquisition(TimeSeries(
            name=f"rsync_pulse_times_{region}",
            description=f"Times of rsync rising edges in {region} (from pyPhotometry pulse_times_1).",
            data=np.ones(len(side_data["pulse_times_1"]), dtype="int8"), unit="n.a.",
            timestamps=np.asarray(side_data["pulse_times_1"], dtype="float64") / 1000.0 + stream_offset_s))

        # Per-sample behavior series at the full 86 Hz photometry clock (length n_samples).
        per_sample_series = {
            f"lick_binary_{region}": ("LickBinary_2.3", "n.a.",
                "Binary lick detection (vdiff threshold 2.3); NaN outside detection window."),
            f"bottle_position_{region}": ("BottlePos", "n.a.",
                "Bottle present (1) / absent (0); NaN outside session window."),
            f"rsync_from_licks_{region}": ("Rsync_aligned-from-licks", "n.a.",
                "rsync signal aligned from the lick data stream."),
            f"rsync_interp_from_video_{region}": ("rSync_interpolated_from_video", "n.a.",
                "rsync interpolated from the video stream."),
        }
        for series_name, (pickle_key, unit, description) in per_sample_series.items():
            behavior_module.add(TimeSeries(
                name=series_name, description=description,
                data=np.asarray(side_data[pickle_key], dtype="float64"),
                unit=unit, rate=SAMPLING_RATE, starting_time=stream_offset_s))

        # Engagement state vectors (auto + manual thresholds)
        for engagement_key in [key for key in side_data if key.startswith("Engagement")]:
            behavior_module.add(TimeSeries(
                name=sanitize(f"{engagement_key}_{region}"),
                description=(f"Engagement state ('{engagement_key}'): "
                            f"animal engaged with spout (1) or not (0), in {region}."),
                data=np.asarray(side_data[engagement_key], dtype="int8"), unit="n.a.",
                rate=SAMPLING_RATE, starting_time=stream_offset_s))

        # Session-cropped series (length n_session_samples), starting at the SessionStart frame.
        cleaned_head_distance = side_data["Cleaned_Head_Distance"]
        n_session_samples = len(cleaned_head_distance)
        burst_vars = side_data["LickBurst_Vars_BurstDefinitionILI_basedThresh2000"]
        distance_states = side_data["Distance_States_Events"]
        cropped_series = {
            f"cumulative_licks_{region}": (burst_vars["CumLicks"], "n.a.", "Cumulative lick count."),
            f"cleaned_head_distance_{region}": (cleaned_head_distance, "pixels",
                "Cleaned head-to-spout distance (interpolated, jump-corrected)."),
            f"distance_state_{region}": (distance_states["state"], "n.a.",
                "Distance state (0/1/2: near/transition/far per QC settings)."),
            f"approach_events_{region}": (distance_states["Approach_events"], "n.a.", "Approach transition events."),
            f"leave_events_{region}": (distance_states["Leave_events"], "n.a.", "Leave transition events."),
        }
        for series_name, (array, unit, description) in cropped_series.items():
            behavior_module.add(TimeSeries(
                name=series_name, description=description, data=np.asarray(array), unit=unit,
                rate=SAMPLING_RATE, starting_time=session_crop_start_s))
        # Labeled burst lick (2D: per-sample burst label)
        behavior_module.add(TimeSeries(
            name=f"labeled_burst_lick_{region}",
            description="Per-sample burst labeling (column 0: in-burst lick flag, column 1: burst id).",
            data=np.asarray(burst_vars["Labeled_BurstLick"], dtype="float64"), unit="n.a.",
            rate=SAMPLING_RATE, starting_time=session_crop_start_s))

        # Derived lick event times (from cumulative-lick increments)
        lick_increments = np.diff(np.asarray(burst_vars["CumLicks"]))
        event_indices = np.where(lick_increments > 0)[0] + 1
        event_times = session_crop_start_s + event_indices / SAMPLING_RATE
        behavior_module.add(TimeSeries(
            name=f"lick_events_{region}",
            description="Detected lick events; data = number of licks registered at each timestamp.",
            data=lick_increments[event_indices - 1].astype("int16"), unit="licks",
            timestamps=event_times.astype("float64")))

        # Lick rate time series (1 s / 1 min / 5 min bins)
        for series_prefix, pickle_key, rate_hz in [("lickrate_1s", "Lickrate_1s", 1.0),
                                                    ("lickrate_1m", "Lickrate_1m", 1.0 / 60.0),
                                                    ("lickrate_5m", "Lickrate_5m", 1.0 / 300.0)]:
            behavior_module.add(TimeSeries(
                name=f"{series_prefix}_{region}",
                description=f"Lick rate in {series_prefix.split('_')[1]} bins (licks/min).",
                data=np.asarray(burst_vars[pickle_key], dtype="float64"), unit="licks/min",
                rate=rate_hz, starting_time=session_crop_start_s))

        # DLC distances + likelihoods (length n_samples, 86 Hz)
        for dlc_key in [key for key in side_data if key.startswith("DLC_")]:
            unit = "pixels" if "Distance" in dlc_key else "probability"
            dlc_module.add(TimeSeries(
                name=sanitize(f"{dlc_key}_{region}"),
                description=f"DeepLabCut '{dlc_key}' in {region}.",
                data=np.asarray(side_data[dlc_key], dtype="float64"), unit=unit,
                rate=SAMPLING_RATE, starting_time=stream_offset_s))

        # Per-lick table (one row per lick)
        n_licks = burst_vars["NumLicks"]
        lick_table_df = pd.DataFrame({
            "lick_duration_ms": np.asarray(burst_vars["LickDurations_ms"], dtype="float64"),
            "interlick_interval_ms": pad_to(burst_vars["InterlickInterval_ms"], n_licks),
            "ili_startend_ms": pad_to(burst_vars["ILI_startend_ms"], n_licks),
        })
        behavior_module.add(DynamicTable.from_dataframe(
            df=lick_table_df, name=f"lick_table_{region}",
            table_description=f"Per-lick durations and inter-lick intervals in {region} ({n_licks} licks)."))

        # Per-burst table (one row per burst)
        n_bursts = burst_vars["NumBursts"]
        burst_table_df = pd.DataFrame({
            "full_burst_duration_ms": np.asarray(burst_vars["Full_BurstDur"], dtype="float64"),
            "lick_burst_duration_ms": np.asarray(burst_vars["Lick_BurstDur"], dtype="float64"),
            "avg_licks_per_burst": np.asarray(burst_vars["Avg_LicksPerBurst"], dtype="float64"),
            "ili_between_bursts_ms": pad_to(burst_vars["ILI_betweenBursts"], n_bursts),
            "ili_within_burst_ms_json": [json_str(np.asarray(within_burst_ilis).tolist())
                                         for within_burst_ilis in burst_vars["ILI_withinBursts"]],
        })
        behavior_module.add(DynamicTable.from_dataframe(
            df=burst_table_df, name=f"burst_table_{region}",
            table_description=(f"Per-burst stats in {region} (burst threshold "
                              f"{burst_vars['BurstThreshold_ms']} ms, {n_bursts} bursts).")))

        # Raw lick data table (full video-frame resolution)
        raw_lick_df = side_data["RawLickData"].copy()
        for column in ("AbsTime", "Abs_time2", "True_Absolute_Time"):
            if column in raw_lick_df.columns:
                raw_lick_df[column] = to_epoch_seconds(raw_lick_df[column])  # epoch seconds (float)
        raw_lick_df = raw_lick_df.astype(
            {column: "float64" for column in raw_lick_df.columns if raw_lick_df[column].dtype == object},
            errors="ignore")
        raw_lick_df["LickFrames_aligned"] = np.asarray(side_data["LickFrames_aligned"], dtype="float64")
        behavior_module.add(DynamicTable.from_dataframe(
            df=raw_lick_df.reset_index(drop=True), name=f"raw_lick_data_{region}",
            table_description=(f"Raw per-frame lick acquisition in {region}. Datetime columns "
                              "(AbsTime, Abs_time2, True_Absolute_Time) are float epoch seconds.")))

        # One metadata row per side (scalars, configs, QC parameters as JSON)
        side_metadata_rows.append({
            "side": side["side"], "com_port": side["com"], "region": region, "hemisphere": side["hemisphere"],
            "hit": side["hit"], "target": side["target"],
            "full_side_name": side_data["Full_side_name"],
            "indicator_470nm": "gACh4h", "indicator_565nm": "rDA3m", "reference_405nm": "gACh4h reference",
            "fiber_coords_ap_ml_dv_mm_json": json_str(list(side["fiber_coords"])),
            "virus_coords_ap_ml_dv_mm_json": json_str([COORDS[region]["ap"],
                                                       side["fiber_coords"][1], COORDS[region]["dv_virus"]]),
            "implant_angle_deg": int(COORDS[region]["angle_deg"]),
            "ppd_filename": side_data["filename"],
            "mode": side_data["mode"], "sampling_rate_hz": float(side_data["sampling_rate"]),
            "led_current_mA_json": json_str(side_data["LED_current"]),
            "volts_per_division_json": json_str(side_data["volts_per_division"]),
            "grams_consumed": float(side_data["GramConsumed"]), "grams_in_pan": float(side_data["GramInPan"]),
            "num_licks": int(burst_vars["NumLicks"]), "num_bursts": int(burst_vars["NumBursts"]),
            "burst_threshold_ms": int(burst_vars["BurstThreshold_ms"]),
            "session_start_frame": int(side_data["SessionStart_frameNum"]),
            "session_end_frame": int(side_data["SessionEnd_frameNum"]),
            "bottle_in_frame": int(side_data["BottleIn_frameNum"]),
            "n_photometry_samples": int(n_samples), "n_session_samples": int(n_session_samples),
            "stream_start_offset_s": float(stream_offset_s),
            "lick_detection_config_json": json_str(side_data["Processing_params"]["Config_Lickdetection"]),
            "hampel_qc_json": json_str(side_data["QC"]["hampel"]),
            "distance_states_qc_json": json_str(side_data["QC"]["distance_states_transition_events"]),
        })

    metadata_module.add(DynamicTable.from_dataframe(
        df=pd.DataFrame(side_metadata_rows), name="session_side_metadata",
        table_description="One row per recording side: scalar metadata, processing config and QC parameters (JSON)."))

    return nwbfile


def convert_one(pkl_path: Path):
    print(f"Reading {pkl_path} ...")
    nwbfile = build_nwb(pkl_path)
    out_path = pkl_path.parent / f"{nwbfile.session_id}.nwb"  # name from session_id
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
    given_paths = [Path(p) for p in sys.argv[1:]]
    pkl_paths = given_paths or sorted(Path(__file__).parent.glob("*_lickprocessed.pkl"))
    if not pkl_paths:
        print("No pickle files given and no *_lickprocessed.pkl found in this directory.")
        return
    for pkl_path in pkl_paths:
        convert_one(pkl_path)


if __name__ == "__main__":
    main()
