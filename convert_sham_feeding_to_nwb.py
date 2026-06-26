"""
One-time conversion of a Berke Lab sham-feeding session (.pkl) to NWB.

The pickle is a 2-tuple of dicts, one per recording hemisphere ("side"). The two hemispheres are
recorded simultaneously, each with its own optical fiber on its own serial port (COM3 / COM4), 
so each recording hemisphere produces one data dict.

Each dict's side/COM/region identity is read from its own 'Full_side_name' field (e.g. 'COM3_Left_mNacSh').

Each side holds pyPhotometry "3EX_2EM_pulsed" data at 86 Hz plus a large stack of
derived behavioral layers (lick detection / bursts / rates, DLC head-to-spout
distances, engagement states, approach/leave events, hampel QC, ...).

PyPhotometry channel mapping:
    analog_1 -> 470 nm -> gACh4h (green ACh sensor)         signal
    analog_2 -> 565 nm -> rDA3m  (red dopamine sensor)      signal
    analog_3 -> 405 nm -> gACh4h (green ACh sensor)         reference (ratiometric)

Style follows https://github.com/calderast/jdb_to_nwb (convert_photometry.py).
"""

import re
import sys
import uuid
import json
import pickle
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


## Metadata for all sessions

TZ = ZoneInfo("America/Los_Angeles")
SAMPLING_RATE = 86.0  # Hz (pyPhotometry sampling rate)

SPECIES = "Rattus norvegicus"
INSTITUTION = "University of California, San Francisco"
LAB = "Berke Lab"
EXPERIMENTER = ["Slomp, Margo"]

EXPERIMENT_DESCRIPTION = (
    "Sham-feeding sucrose task with bilateral nucleus accumbens fiber photometry. "
    "Two fibers record the green acetylcholine sensor gACh4h (470 nm, with 405 nm "
    "reference channel) and the red dopamine sensor rDA3m (565 nm, no reference channel) from two "
    "nucleus accumbens sites (core and/or medial shell; see the 'surgery' field for this session's "
    "targets), while licking at a sucrose spout is detected and the animal's head/nose distance to "
    "the spout is tracked. "
    "Task structure: 10 min baseline (no bottle available), then 60 x 30 s spout-access periods each "
    "followed by 30 s of no access. A 350 ms 4000 Hz tone marks each trial start. The animal is sham-fed "
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
# DV is .2mm higher for the fiber tip (recording site) vs the virus injection site
COORDS = {
    "NacCore": dict(ap=1.7, ml=1.7, dv_fiber=6.8, dv_virus=7.0, angle_deg=0),
    "mNacSh":  dict(ap=1.3, ml=1.6, dv_fiber=6.2, dv_virus=6.4, angle_deg=6),
}

# Readable region names for the surgery description
REGION_DISPLAY = {"NacCore": "NAc core", "mNacSh": "medial NAc shell"}

# per pyPhotometry channel: (pickle key, excitation wavelength nm, hampel-cleaned key, sensor, description)
CHANNELS = [
    ("analog_1", 470, "analog1_hampel", "gACh4h", "gACh4h signal"),
    ("analog_2", 565, "analog2_hampel", "rDA3m",  "rDA3m signal"),
    ("analog_3", 405, "analog3_hampel", "gACh4h", "gACh4h reference (ratiometric)"),
]


## Helpers

def sanitize(name: str) -> str:
    """Make a string safe/readable for an NWB object name."""
    name = name.replace(".", "p") # "." isn't allowed in nwb names so use a "p" instead (e.g. "274.64" -> "274p64")
    name = re.sub(r"[^0-9A-Za-z]+", "_", name) # collapse any other run of illegal chars (/, :, -, space) to one "_"
    return name.strip("_")


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

    The region part may carry a qualifier when the target was not confirmed by histology 
    (e.g. 'COM3_Left_Potential_NacCore', with Target_* = 'Potential_NacCore' and Hit_* = 'Unknown').

    Hit_* / Target_* and the COM ports are read from the matching L/R field (i.e. Target_R or Target_L).
    """
    # Parse Full_side_name to get port (COM3 or COM4), hemisphere (Left or Right), and region descriptor
    name_parts = str(side_data["Full_side_name"]).split("_")
    com_port, side_name, region_descriptor = name_parts[0], name_parts[1], "_".join(name_parts[2:])
    hemisphere = side_name.lower()  # "Left" -> "left", "Right" -> "right"

    # Canonical brain region for coordinate lookup and naming. The descriptor may carry a qualifier
    # (e.g. "Potential_NacCore"), so match the known region as a suffix.
    region = next((known for known in COORDS if region_descriptor.endswith(known)), region_descriptor)
    # Complain if we get confused
    assert region in COORDS, (
        f"Unrecognized region '{region_descriptor}' in Full_side_name "
        f"'{side_data['Full_side_name']}' (known regions: {list(COORDS)})")

    # Get region targeted (based on coords) and actually hit (verified via histology) for this hemisphere
    target = side_data["Target_L"] if side_name == "Left" else side_data["Target_R"]
    hit = side_data["Hit_L"] if side_name == "Left" else side_data["Hit_R"]
    # Create `label` (e.g. "left_mNacSh") to name this side in NWB objects
    label = f"{hemisphere}_{region}"
    return dict(index=index, side=side_name, com=com_port, region=region,
                hemisphere=hemisphere, hit=hit, target=target, label=label)


def build_surgery(sides):
    """Build the surgery description for the nwb based on the regions targeted this session.

    Only the regions present in `sides` are described, with coordinates for core/shell pulled from COORDS.
    Reports the ML sign(s) for the hemisphere(s) actually implanted (negative for left hemisphere).
    """
    hemispheres_by_region = {}  # region -> set of hemispheres
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
    # Two simultaneously-recorded hemispheres -> a 2-tuple of dicts (one dict per hemisphere/side).
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

    # One optical fiber per recording side, with both indicators (gACh4h, rDA3m) injected at each site.
    # Keyed by side label (hemisphere+region)
    fibers_by_side = {}
    indicators_by_side_and_sensor = {}
    for side in sides:
        # Region is "NacCore" or "mNacSh", hemisphere is "left" or "right", label is e.g. "left_mNacSh"
        region, hemisphere, side_label = side["region"], side["hemisphere"], side["label"]
        # Get virus and fiber coords for this region
        coords = COORDS[region]
        ml_mm = -coords["ml"] if hemisphere == "left" else coords["ml"]
        fiber_coords = (coords["ap"], ml_mm, coords["dv_fiber"])  # AP, ML, DV (mm) of the fiber tip
        virus_coords = (coords["ap"], ml_mm, coords["dv_virus"])  # AP, ML, DV (mm) of the virus injection
        side["fiber_coords"] = fiber_coords
        side["virus_coords"] = virus_coords

        angle_label = "no angle" if coords["angle_deg"] == 0 else f"{coords['angle_deg']} degree angle"
        fiber = OpticalFiber(
            name=f"Doric 200um 10mm Optic Fiber ({hemisphere} {region})",
            manufacturer="Doric", model="MFC_200/250-0.66_10mm_MF2.5_FLT",
            numerical_aperture=0.66, core_diameter_in_um=200.0,
            description=(f"Doric 200 um fiber (B280-2615-10) implanted in {hemisphere} {region} at "
                         f"AP {fiber_coords[0]}, ML {fiber_coords[1]}, DV {fiber_coords[2]} mm "
                         f"from dura ({angle_label})."))
        nwbfile.add_device(fiber)
        fibers_by_side[side_label] = fiber

        for sensor in ("gACh4h", "rDA3m"):
            indicator_info = INDICATOR_INFO[sensor]
            indicator = Indicator(
                name=f"{sensor} ({hemisphere} {region})",
                label=indicator_info["label"], description=indicator_info["description"],
                manufacturer=indicator_info["manufacturer"], injection_location=region,
                injection_coordinates_in_mm=virus_coords)
            nwbfile.add_device(indicator)
            indicators_by_side_and_sensor[(side_label, sensor)] = indicator

    # Fiber photometry table: one row per (side, channel)
    fiber_table = FiberPhotometryTable(
        name="fiber_photometry_table",
        description="Fiber, indicator and excitation source for each recorded channel.")
    row_index_by_channel = {}  # (side_label, analog_key) -> row index
    next_row_index = 0
    for side in sides:
        region, side_label = side["region"], side["label"]
        for analog_key, wavelength, _hampel_key, sensor, _role in CHANNELS:
            fiber_table.add_row(
                location=region,
                coordinates=side["fiber_coords"],  # AP, ML, DV (mm) of the recording fiber tip
                optical_fiber=fibers_by_side[side_label],
                photodetector=photodetector,
                dichroic_mirror=dichroic_mirror,
                indicator=indicators_by_side_and_sensor[(side_label, sensor)],
                excitation_source=excitation_sources[wavelength],
            )
            row_index_by_channel[(side_label, analog_key)] = next_row_index
            next_row_index += 1
    nwbfile.add_lab_meta_data(FiberPhotometry(name="fiber_photometry", fiber_photometry_table=fiber_table))

    def channel_table_region(analog_key, side_label):
        """A single-row FiberPhotometryTableRegion pointing at this channel's table row."""
        return fiber_table.create_fiber_photometry_table_region(
            region=[row_index_by_channel[(side_label, analog_key)]], description=f"{analog_key} @ {side_label}")

    # Per-side signals and tables.
    # Each iteration handles one recording hemisphere's dict (its own photometry + derived layers)
    side_metadata_rows = []
    for side in sides:
        side_data = sides_data[side["index"]]   # the raw pickle dict for this hemisphere
        region = side["region"]
        side_label = side["label"]                       # e.g. "left_mNacSh", for NWB name suffix
        side_desc = f"{side['hemisphere']} {region}"     # e.g. "left mNacSh", for descriptions
        n_samples = len(side_data["analog_1"])

        # NWB stores each regularly-sampled series as (starting_time, rate).
        # So sample i is at starting_time + i / SAMPLING_RATE. 
        # Each dict has two 86 Hz timebases:
        #   - the full photometry stream (length n_samples) starts at session_start (starting_time=0);
        #   - the session-cropped arrays (length n_session_samples) trim that stream to the actual session
        #     window, so they start at sample SessionStart_frameNum (e.g. ~2451 -> ~28.5 s in).
        #
        # We anchor BOTH sides at session_start=0 (the shared rsync pulses show the two sides begin within
        # ~1 sample of each other). Within a side every stream shares one clock, so 86 Hz placement is
        # exact. Across sides the clocks drift ~50 ms over a session, left for downstream correction
        # using the rsync_pulse_times_* series stored in acquisition.
        session_crop_start_s = int(side_data["SessionStart_frameNum"]) / SAMPLING_RATE

        # Add each raw, pyPhotometry-filtered, and hampel-cleaned series as a FiberPhotometryResponseSeries
        for analog_key, wavelength, hampel_key, _sensor, role in CHANNELS:
            # Raw signal
            nwbfile.add_acquisition(FiberPhotometryResponseSeries(
                name=f"raw_{wavelength}_{side_label}",
                description=f"Raw {role} ({wavelength} nm) in {side_desc}. pyPhotometry {side_data['mode']}.",
                data=np.asarray(side_data[analog_key], dtype="float64"),
                unit="V", rate=SAMPLING_RATE, starting_time=0.0,
                fiber_photometry_table_region=channel_table_region(analog_key, side_label)))
            # Filtered signal
            nwbfile.add_acquisition(FiberPhotometryResponseSeries(
                name=f"filt_{wavelength}_{side_label}",
                description=f"pyPhotometry-filtered {role} ({wavelength} nm) in {side_desc}.",
                data=np.asarray(side_data[f"{analog_key}_filt"], dtype="float64"),
                unit="V", rate=SAMPLING_RATE, starting_time=0.0,
                fiber_photometry_table_region=channel_table_region(analog_key, side_label)))
            # Hampel-filtered signal
            nwbfile.add_acquisition(FiberPhotometryResponseSeries(
                name=f"hampel_{wavelength}_{side_label}",
                description=(f"Hampel-filtered {role} ({wavelength} nm) in {side_desc} "
                            f"(window {side_data['QC']['hampel']['window_sec']}s, "
                            f"{side_data['QC']['hampel']['n_sigmas']} sigma)."),
                data=np.asarray(side_data[hampel_key], dtype="float64"),
                unit="V", rate=SAMPLING_RATE, starting_time=0.0,
                fiber_photometry_table_region=channel_table_region(analog_key, side_label)))

        # Add digital sync + rsync pulse times
        nwbfile.add_acquisition(TimeSeries(
            name=f"digital_sync_{side_label}", description=f"Digital sync input (rsync) in {side_desc}.",
            data=np.asarray(side_data["digital_1"], dtype="int8"), unit="n.a.",
            rate=SAMPLING_RATE, starting_time=0.0))
        nwbfile.add_acquisition(TimeSeries(
            name=f"rsync_pulse_times_{side_label}",
            description=f"Times of rsync rising edges in {side_desc} (from pyPhotometry pulse_times_1).",
            data=np.ones(len(side_data["pulse_times_1"]), dtype="int8"), unit="n.a.",
            timestamps=np.asarray(side_data["pulse_times_1"], dtype="float64") / 1000.0))

        # Per-sample behavior series at the full 86 Hz photometry clock (length n_samples).
        per_sample_series = {
            f"lick_binary_{side_label}": ("LickBinary_2.3", "n.a.",
                "Binary lick detection (vdiff threshold 2.3); NaN outside detection window."),
            f"bottle_position_{side_label}": ("BottlePos", "n.a.",
                "Bottle present (1) / absent (0); NaN outside session window."),
            f"rsync_from_licks_{side_label}": ("Rsync_aligned-from-licks", "n.a.",
                "rsync signal aligned from the lick data stream."),
            f"rsync_interp_from_video_{side_label}": ("rSync_interpolated_from_video", "n.a.",
                "rsync interpolated from the video stream."),
        }
        for series_name, (pickle_key, unit, description) in per_sample_series.items():
            behavior_module.add(TimeSeries(
                name=series_name, description=description,
                data=np.asarray(side_data[pickle_key], dtype="float64"),
                unit=unit, rate=SAMPLING_RATE, starting_time=0.0))

        # Engagement state vectors (auto + manual thresholds)
        for engagement_key in [key for key in side_data if key.startswith("Engagement")]:
            behavior_module.add(TimeSeries(
                name=sanitize(f"{engagement_key}_{side_label}"),
                description=(f"Engagement state ('{engagement_key}'): "
                            f"animal engaged with spout (1) or not (0), in {side_desc}."),
                data=np.asarray(side_data[engagement_key], dtype="int8"), unit="n.a.",
                rate=SAMPLING_RATE, starting_time=0.0))

        # Session-cropped series (length n_session_samples), starting at the SessionStart frame.
        cleaned_head_distance = side_data["Cleaned_Head_Distance"]
        n_session_samples = len(cleaned_head_distance)
        burst_vars = side_data["LickBurst_Vars_BurstDefinitionILI_basedThresh2000"]
        distance_states = side_data["Distance_States_Events"]
        cropped_series = {
            f"cumulative_licks_{side_label}": (burst_vars["CumLicks"], "n.a.", "Cumulative lick count."),
            f"cleaned_head_distance_{side_label}": (cleaned_head_distance, "pixels",
                "Cleaned head-to-spout distance (interpolated, jump-corrected)."),
            f"distance_state_{side_label}": (distance_states["state"], "n.a.",
                "Distance state (0/1/2: near/transition/far per QC settings)."),
            f"approach_events_{side_label}": (distance_states["Approach_events"], "n.a.", "Approach transition events."),
            f"leave_events_{side_label}": (distance_states["Leave_events"], "n.a.", "Leave transition events."),
        }
        for series_name, (array, unit, description) in cropped_series.items():
            behavior_module.add(TimeSeries(
                name=series_name, description=description, data=np.asarray(array), unit=unit,
                rate=SAMPLING_RATE, starting_time=session_crop_start_s))
        # Labeled burst lick (2D: per-sample burst label)
        behavior_module.add(TimeSeries(
            name=f"labeled_burst_lick_{side_label}",
            description="Per-sample burst labeling (column 0: in-burst lick flag, column 1: burst id).",
            data=np.asarray(burst_vars["Labeled_BurstLick"], dtype="float64"), unit="n.a.",
            rate=SAMPLING_RATE, starting_time=session_crop_start_s))

        # Add derived lick event times (from cumulative-lick increments): 
        # CumLicks is a running total, so a positive diff marks the sample(s) where new licks 
        # were registered. Here we convert those sample indices to lick timestamps.
        lick_increments = np.diff(np.asarray(burst_vars["CumLicks"]))
        event_indices = np.where(lick_increments > 0)[0] + 1
        event_times = session_crop_start_s + event_indices / SAMPLING_RATE
        behavior_module.add(TimeSeries(
            name=f"lick_events_{side_label}",
            description="Detected lick events; data = number of licks registered at each timestamp.",
            data=lick_increments[event_indices - 1].astype("int16"), unit="licks",
            timestamps=event_times.astype("float64")))

        # Lick rate time series (1 s / 1 min / 5 min bins)
        for series_prefix, pickle_key, rate_hz in [("lickrate_1s", "Lickrate_1s", 1.0),
                                                    ("lickrate_1m", "Lickrate_1m", 1.0 / 60.0),
                                                    ("lickrate_5m", "Lickrate_5m", 1.0 / 300.0)]:
            behavior_module.add(TimeSeries(
                name=f"{series_prefix}_{side_label}",
                description=f"Lick rate in {series_prefix.split('_')[1]} bins (licks/min).",
                data=np.asarray(burst_vars[pickle_key], dtype="float64"), unit="licks/min",
                rate=rate_hz, starting_time=session_crop_start_s))

        # DLC distances + likelihoods (length n_samples, 86 Hz)
        for dlc_key in [key for key in side_data if key.startswith("DLC_")]:
            unit = "pixels" if "Distance" in dlc_key else "probability"
            dlc_module.add(TimeSeries(
                name=sanitize(f"{dlc_key}_{side_label}"),
                description=f"DeepLabCut '{dlc_key}' in {side_desc}.",
                data=np.asarray(side_data[dlc_key], dtype="float64"), unit=unit,
                rate=SAMPLING_RATE, starting_time=0.0))

        # Per-lick table (one row per lick). 
        # There are n_licks durations but only n_licks-1 inter-lick intervals, 
        # so the interval columns are NaN-padded out to n_licks.
        n_licks = burst_vars["NumLicks"]
        lick_table_df = pd.DataFrame({
            "lick_duration_ms": np.asarray(burst_vars["LickDurations_ms"], dtype="float64"),
            "interlick_interval_ms": pad_to(burst_vars["InterlickInterval_ms"], n_licks),
            "ili_startend_ms": pad_to(burst_vars["ILI_startend_ms"], n_licks),
        })
        behavior_module.add(DynamicTable.from_dataframe(
            df=lick_table_df, name=f"lick_table_{side_label}",
            table_description=f"Per-lick durations and inter-lick intervals in {side_desc} ({n_licks} licks)."))

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
            df=burst_table_df, name=f"burst_table_{side_label}",
            table_description=(f"Per-burst stats in {side_desc} (burst threshold "
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
            df=raw_lick_df.reset_index(drop=True), name=f"raw_lick_data_{side_label}",
            table_description=(f"Raw per-frame lick acquisition in {side_desc}. Datetime columns "
                              "(AbsTime, Abs_time2, True_Absolute_Time) are float epoch seconds.")))

        # One metadata row per side (scalars, configs, QC parameters as JSON)
        side_metadata_rows.append({
            "side_label": side_label,
            "side": side["side"], "com_port": side["com"], "region": region, "hemisphere": side["hemisphere"],
            "hit": side["hit"], "target": side["target"],
            "full_side_name": side_data["Full_side_name"],
            "indicator_470nm": "gACh4h", "indicator_565nm": "rDA3m", "reference_405nm": "gACh4h reference",
            "fiber_coords_ap_ml_dv_mm_json": json_str(list(side["fiber_coords"])),
            "virus_coords_ap_ml_dv_mm_json": json_str(list(side["virus_coords"])),
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
            "lick_detection_config_json": json_str(side_data["Processing_params"]["Config_Lickdetection"]),
            "hampel_qc_json": json_str(side_data["QC"]["hampel"]),
            "distance_states_qc_json": json_str(side_data["QC"]["distance_states_transition_events"]),
        })

    metadata_module.add(DynamicTable.from_dataframe(
        df=pd.DataFrame(side_metadata_rows), name="session_side_metadata",
        table_description="One row per recording side: scalar metadata, processing config and QC parameters (JSON)."))

    return nwbfile


## Quality control figures + text inventory (written per NWB)
#
# All QC outputs are read back from the written NWB (not the source pickle) -- the point of QC is to
# verify what actually landed in the file.

# Plot styling for the QC figures.
WAVELENGTH_COLORS = {470: "#2CA02C", 565: "#D62728", 405: "#8FBF8F"}  # gACh4h / rDA3m / 405 reference
WAVELENGTH_SENSOR = {470: "gACh4h", 565: "rDA3m", 405: "gACh4h ref"}
PHOTOMETRY_SUBPLOT_ORDER = (470, 405, 565)  # gACh4h signal, its 405 reference, then rDA3m


def time_vector(timeseries):
    """Seconds for a TimeSeries, whether it stores a rate or explicit timestamps."""
    if timeseries.timestamps is not None:
        return np.asarray(timeseries.timestamps[:])
    n = timeseries.data.shape[0]
    start = timeseries.starting_time if timeseries.starting_time is not None else 0.0
    return start + np.arange(n) / timeseries.rate


def write_inventory(nwb, nwb_name, txt_path):
    """Write a human-readable inventory of everything in the NWB to a .txt file."""
    lines = ["=" * 80, f"NWB INVENTORY -- {nwb_name}", "=" * 80]
    subject = nwb.subject
    lines += [
        f"session_id:         {nwb.session_id}",
        f"identifier:         {nwb.identifier}",
        f"session_start_time: {nwb.session_start_time}",
        f"experimenter:       {nwb.experimenter}",
        f"institution / lab:  {nwb.institution} / {nwb.lab}",
        f"subject:            {subject.subject_id} | {subject.species} | sex {subject.sex} | "
        f"genotype {subject.genotype} | strain {subject.strain} | DOB {subject.date_of_birth}",
        f"surgery:            {nwb.surgery}",
        f"virus:              {nwb.virus}",
        f"notes:              {nwb.notes}",
    ]

    lines += ["", "ACQUISITION:"]
    for name, obj in nwb.acquisition.items():
        lines.append(f"  {name:34s} {type(obj).__name__}  shape={getattr(obj.data, 'shape', '?')}  "
                     f"unit={getattr(obj, 'unit', '')}")

    for module_name in nwb.processing:
        lines += ["", f"PROCESSING / {module_name}:"]
        for name, obj in nwb.processing[module_name].data_interfaces.items():
            if hasattr(obj, "to_dataframe"):
                table = obj.to_dataframe()
                lines.append(f"  {name:42s} DynamicTable  rows={len(table)}  cols={list(table.columns)}")
            else:
                lines.append(f"  {name:42s} {type(obj).__name__}  shape={getattr(obj.data, 'shape', '?')}")

    lines += ["", "DEVICES:"]
    for name in nwb.devices:
        lines.append(f"  {name}")

    fiber_table = nwb.get_lab_meta_data("fiber_photometry").fiber_photometry_table.to_dataframe()
    lines += ["", "FIBER PHOTOMETRY TABLE:"]
    for row_idx, row in fiber_table.iterrows():
        lines.append(f"  row{row_idx}: location={row['location']} "
                     f"coords(AP,ML,DV)={list(np.round(np.asarray(row['coordinates']), 2))} "
                     f"indicator={row['indicator'].name} excitation={row['excitation_source'].name}")

    side_metadata = nwb.processing["session_metadata"]["session_side_metadata"].to_dataframe()
    lines += ["", "SESSION SIDE METADATA:", side_metadata.T.to_string()]

    txt_path.write_text("\n".join(lines))


def save_qc_outputs(nwb, output_dir):
    """Write per-NWB QC figures and a text inventory into a `{session_id}_figures/` folder.

    Everything is read from the NWB `nwb` (the written file). Figures mirror explore_pickle.ipynb;
    every saved figure carries the NWB filename in its suptitle. Per-side figures are saved once per
    recording side; the trial / engagement / state / sync figures stack both sides as vertical subplots.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nwb_name = f"{nwb.session_id}.nwb"
    fig_dir = output_dir / f"{nwb.session_id}_figures"
    fig_dir.mkdir(exist_ok=True)

    behavior = nwb.processing["behavior"]
    dlc = nwb.processing["dlc"]
    side_metadata = nwb.processing["session_metadata"]["session_side_metadata"].to_dataframe()
    sides = [(row["side_label"], row["full_side_name"]) for _, row in side_metadata.iterrows()]

    def save(fig, name, description, side_name):
        fig.suptitle(f"{nwb_name}   --   {description}   --   {side_name}")
        fig.tight_layout()
        fig.savefig(fig_dir / name, dpi=110, bbox_inches="tight")
        plt.close(fig)

    # ------- Per-side figures -------
    for side_label, side_name in sides:
        # Photometry: one figure per processing level, one subplot per wavelength
        for level, prefix in [("raw", "raw"), ("filtered", "filt"), ("hampel", "hampel")]:
            fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
            for axis, wavelength in zip(axes, PHOTOMETRY_SUBPLOT_ORDER):
                series = nwb.acquisition[f"{prefix}_{wavelength}_{side_label}"]
                axis.plot(time_vector(series) / 60, series.data[:], color=WAVELENGTH_COLORS[wavelength], lw=0.6)
                axis.set_ylabel("V"); axis.set_title(f"{wavelength} nm ({WAVELENGTH_SENSOR[wavelength]})", fontsize=9)
            axes[-1].set_xlabel("time (min)")
            save(fig, f"photometry_{level}_{side_name}.png", f"{level} photometry", side_name)

        # Licking: cumulative licks + lick rates (1 s / 1 min / 5 min)
        cumulative = behavior[f"cumulative_licks_{side_label}"]
        fig, axes = plt.subplots(4, 1, figsize=(13, 9))
        axes[0].plot(time_vector(cumulative) / 60, cumulative.data[:]); axes[0].set_ylabel("cum licks")
        axes[0].set_title("Cumulative licks"); axes[0].set_xlabel("time (min)")
        for axis, (rate_key, label) in zip(axes[1:], [("lickrate_1s", "1 s"), ("lickrate_1m", "1 min"),
                                                      ("lickrate_5m", "5 min")]):
            rate = behavior[f"{rate_key}_{side_label}"]
            axis.plot(time_vector(rate) / 60, rate.data[:], lw=0.8)
            axis.set_ylabel("licks/min"); axis.set_title(f"Lick rate ({label} bins)"); axis.set_xlabel("time (min)")
        save(fig, f"licks_{side_name}.png", "licking overview", side_name)

        # Burst structure: from the lick and burst tables
        burst_df = behavior[f"burst_table_{side_label}"].to_dataframe()
        lick_df = behavior[f"lick_table_{side_label}"].to_dataframe()
        within_burst_ilis = [v for s in burst_df["ili_within_burst_ms_json"] for v in json.loads(s)]
        fig, axes = plt.subplots(2, 2, figsize=(13, 7))
        axes[0, 0].hist(lick_df["lick_duration_ms"].dropna(), bins=80); axes[0, 0].set_title("Lick durations (ms)"); axes[0, 0].set_xlim(0, 200)
        axes[0, 1].hist(burst_df["avg_licks_per_burst"], bins=30); axes[0, 1].set_title("Licks per burst")
        axes[1, 0].hist(burst_df["full_burst_duration_ms"] / 1000, bins=30, alpha=0.6, label="full burst")
        axes[1, 0].hist(burst_df["lick_burst_duration_ms"] / 1000, bins=30, alpha=0.6, label="lick burst")
        axes[1, 0].set_title("Burst durations (s)"); axes[1, 0].legend(fontsize=8)
        if within_burst_ilis:
            axes[1, 1].hist(np.clip(within_burst_ilis, 0, 1000), bins=60, alpha=0.6, label="within-burst ILI")
        axes[1, 1].hist(np.clip(burst_df["ili_between_bursts_ms"].dropna(), 0, 60000), bins=60, alpha=0.6, label="between-burst ILI")
        axes[1, 1].set_title("ILI within vs between bursts (ms)"); axes[1, 1].legend(fontsize=8)
        save(fig, f"bursts_{side_name}.png", "burst structure", side_name)

        # DLC distances + likelihoods
        fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
        for dist_key, label in [("DLC_Distance_head_middle_spout", "head_middle-spout"),
                                ("DLC_Distance_nose_spout", "nose-spout"),
                                ("DLC_Distance_head_middle_nose", "head-nose")]:
            series = dlc[f"{dist_key}_{side_label}"]
            axes[0].plot(time_vector(series) / 60, series.data[:], lw=0.6, label=label)
        axes[0].set_ylabel("distance (px)"); axes[0].legend(fontsize=8); axes[0].set_title("DLC distances")
        for like_key, label in [("DLC_Likelihood_head_middle", "head_middle"), ("DLC_Likelihood_nose", "nose")]:
            series = dlc[f"{like_key}_{side_label}"]
            axes[1].plot(time_vector(series) / 60, series.data[:], lw=0.6, label=label)
        axes[1].set_ylabel("likelihood"); axes[1].set_xlabel("time (min)"); axes[1].legend(fontsize=8); axes[1].set_title("DLC likelihoods")
        save(fig, f"dlc_{side_name}.png", "DLC tracking", side_name)

        # QC: largest raw-vs-hampel correction across channels
        worst_wavelength, worst_idx, worst_correction = 470, 0, -1.0
        for wavelength in (470, 565, 405):
            raw_values = nwb.acquisition[f"raw_{wavelength}_{side_label}"].data[:]
            hampel_values = nwb.acquisition[f"hampel_{wavelength}_{side_label}"].data[:]
            correction = np.abs(raw_values - hampel_values)
            peak_idx = int(np.argmax(correction))
            if correction[peak_idx] > worst_correction:
                worst_wavelength, worst_idx, worst_correction = wavelength, peak_idx, correction[peak_idx]
        raw_series = nwb.acquisition[f"raw_{worst_wavelength}_{side_label}"]
        hampel_series = nwb.acquisition[f"hampel_{worst_wavelength}_{side_label}"]
        times = time_vector(raw_series)
        lo, hi = max(0, worst_idx - 200), worst_idx + 200
        fig, axis = plt.subplots(figsize=(13, 3))
        axis.plot(times[lo:hi], raw_series.data[lo:hi], label=f"raw {worst_wavelength} nm", lw=0.8)
        axis.plot(times[lo:hi], hampel_series.data[lo:hi], label="hampel", lw=0.8)
        axis.set_xlabel("time (s)"); axis.set_ylabel("V"); axis.legend(fontsize=8)
        axis.set_title(f"Largest raw-vs-hampel correction ({worst_wavelength} nm)")
        save(fig, f"qc_hampel_{side_name}.png", "QC hampel", side_name)

        # Combined session overview: 5 min mid-session
        rda = nwb.acquisition[f"hampel_565_{side_label}"]
        ach = nwb.acquisition[f"hampel_470_{side_label}"]
        lick_binary = behavior[f"lick_binary_{side_label}"]
        bottle = behavior[f"bottle_position_{side_label}"]
        state = behavior[f"distance_state_{side_label}"]
        state_t_min = time_vector(state) / 60
        auto_engagement_key = [k for k in behavior.data_interfaces if k.startswith("Engagement")
                               and "auto" in k and "head" in k and k.endswith(side_label)][0]
        auto_engagement = behavior[auto_engagement_key]
        full_t_min = time_vector(rda) / 60
        window_start = full_t_min[-1] / 2
        window_end = min(window_start + 5, full_t_min[-1])
        in_window = (full_t_min >= window_start) & (full_t_min <= window_end)
        state_in_window = (state_t_min >= window_start) & (state_t_min <= window_end)
        fig, axes = plt.subplots(6, 1, figsize=(13, 10), sharex=True)
        axes[0].plot(full_t_min[in_window], rda.data[:][in_window], color="#D62728", lw=0.7); axes[0].set_ylabel("rDA (V)")
        axes[1].plot(full_t_min[in_window], ach.data[:][in_window], color="#2CA02C", lw=0.7); axes[1].set_ylabel("gACh4h (V)")
        axes[2].fill_between(full_t_min[in_window], 0, np.nan_to_num(lick_binary.data[:])[in_window], step="mid", color="k"); axes[2].set_ylabel("lick")
        axes[3].fill_between(full_t_min[in_window], 0, np.nan_to_num(bottle.data[:])[in_window], step="mid", color="tab:blue"); axes[3].set_ylabel("bottle")
        axes[4].fill_between(full_t_min[in_window], 0, auto_engagement.data[:][in_window], step="mid", color="tab:orange"); axes[4].set_ylabel("engaged")
        axes[5].plot(state_t_min[state_in_window], state.data[:][state_in_window], lw=0.8, color="tab:purple"); axes[5].set_ylabel("state")
        axes[5].set_xlabel("time (min)")
        save(fig, f"session_overview_{side_name}.png",
             f"session overview ({window_start:.0f}-{window_end:.0f} min)", side_name)

    # ------- Both-sides figures (one vertical subplot per side) -------
    n_sides = len(sides)

    # rsync inter-pulse interval, one subplot per side. Bin on the 86 Hz sample grid (one bin per
    # integer sample gap = 1000/86 ms) and share bins across sides, so each box's pulses land on the
    # same bins and the +/-1-sample digitization jitter is directly comparable between sides.
    sample_ms = 1000.0 / SAMPLING_RATE
    inter_pulse_by_side = {
        side_label: np.diff(np.asarray(nwb.acquisition[f"rsync_pulse_times_{side_label}"].timestamps[:]) * 1000)
        for side_label, _ in sides
    }
    gaps_in_samples = np.concatenate([np.round(ipi / sample_ms) for ipi in inter_pulse_by_side.values()])
    bin_edges = (np.arange(gaps_in_samples.min(), gaps_in_samples.max() + 2) - 0.5) * sample_ms
    fig, axes = plt.subplots(n_sides, 1, figsize=(13, 3 * n_sides), sharex=True)
    for axis, (side_label, side_name) in zip(np.atleast_1d(axes), sides):
        inter_pulse_ms = inter_pulse_by_side[side_label]
        axis.hist(inter_pulse_ms, bins=bin_edges)
        axis.set_ylabel("count")
        axis.set_title(f"{side_name}: rsync inter-pulse interval, n={len(inter_pulse_ms)}", fontsize=9)
    np.atleast_1d(axes)[-1].set_xlabel("ms (binned on the 86 Hz sample grid)")
    save(fig, "sync.png", "rsync inter-pulse interval", "both sides")

    # Bottle / trial structure, one subplot per side
    fig, axes = plt.subplots(n_sides, 1, figsize=(13, 2.6 * n_sides), sharex=True)
    for axis, (side_label, side_name) in zip(np.atleast_1d(axes), sides):
        bottle = behavior[f"bottle_position_{side_label}"]
        bottle_t_min = time_vector(bottle) / 60
        bottle_in = np.nan_to_num(bottle.data[:]) > 0.5
        access_starts = np.where(bottle_in[1:] & ~bottle_in[:-1])[0] + 1
        access_ends = np.where(~bottle_in[1:] & bottle_in[:-1])[0] + 1
        n_access = min(len(access_starts), len(access_ends))
        for start_idx, end_idx in zip(access_starts[:n_access], access_ends[:n_access]):
            axis.axvspan(bottle_t_min[start_idx], bottle_t_min[end_idx], color="tab:blue", alpha=0.25, lw=0)
        axis.set_xlim(bottle_t_min[0], bottle_t_min[-1]); axis.set_ylim(0, 1); axis.set_yticks([])
        axis.set_title(f"{side_name}: {len(access_starts)} access periods", fontsize=9)
    np.atleast_1d(axes)[-1].set_xlabel("time (min)")
    save(fig, "bottle_trials.png", "bottle/trial structure", "both sides")

    # Engagement rasters, one subplot per side
    fig, axes = plt.subplots(n_sides, 1, figsize=(13, 4 * n_sides), sharex=True)
    for axis, (side_label, side_name) in zip(np.atleast_1d(axes), sides):
        engagement_keys = sorted(k for k in behavior.data_interfaces
                                 if k.startswith("Engagement") and k.endswith(side_label))
        for row, key in enumerate(engagement_keys):
            series = behavior[key]
            axis.fill_between(time_vector(series) / 60, row, row + series.data[:] * 0.9, step="mid", lw=0)
        axis.set_yticks(np.arange(len(engagement_keys)) + 0.45); axis.set_yticklabels(engagement_keys, fontsize=6)
        axis.set_title(side_name, fontsize=9)
    np.atleast_1d(axes)[-1].set_xlabel("time (min)")
    save(fig, "engagement.png", "engagement vectors", "both sides")

    # Distance state + approach/leave transitions, one subplot per side
    fig, axes = plt.subplots(n_sides, 1, figsize=(13, 2.8 * n_sides), sharex=True)
    for axis, (side_label, side_name) in zip(np.atleast_1d(axes), sides):
        state = behavior[f"distance_state_{side_label}"]
        state_t_min = time_vector(state) / 60
        approach = behavior[f"approach_events_{side_label}"]
        leave = behavior[f"leave_events_{side_label}"]
        approach_times = (time_vector(approach) / 60)[np.asarray(approach.data[:]) > 0]
        leave_times = (time_vector(leave) / 60)[np.asarray(leave.data[:]) > 0]
        axis.plot(state_t_min, state.data[:], lw=0.6)
        axis.plot(approach_times, np.full_like(approach_times, 2.1), "^", ms=4, color="g", label="approach")
        axis.plot(leave_times, np.full_like(leave_times, -0.1), "v", ms=4, color="r", label="leave")
        axis.set_ylabel("state"); axis.set_title(side_name, fontsize=9); axis.legend(fontsize=7)
    np.atleast_1d(axes)[-1].set_xlabel("time (min)")
    save(fig, "distance_states.png", "distance states", "both sides")

    write_inventory(nwb, nwb_name, fig_dir / f"{nwb.session_id}_inventory.txt")
    print(f"Saved QC figures + inventory to {fig_dir}")

def convert_one(pkl_path: Path):
    print(f"Reading {pkl_path} ...")
    nwbfile = build_nwb(pkl_path)
    out_path = pkl_path.parent / f"{nwbfile.session_id}.nwb"  # name from session_id
    print(f"Writing {out_path} ...")
    with NWBHDF5IO(out_path, mode="w") as io:
        io.write(nwbfile)

    # Re-read the written file and run QC on it, so the figures + inventory reflect the NWB itself.
    with NWBHDF5IO(out_path, mode="r") as io:
        nwb = io.read()
        print(f"Done. Re-read OK: {nwb.session_id} | "
              f"acquisition: {len(nwb.acquisition)}, "
              f"behavior: {len(nwb.processing['behavior'].data_interfaces)}, "
              f"dlc: {len(nwb.processing['dlc'].data_interfaces)}")
        save_qc_outputs(nwb, out_path.parent)


def main():
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
