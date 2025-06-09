from io import BytesIO
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Any
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import pretty_midi
import pandas as pd
import bisect
from fractions import Fraction
import warnings

FRENCH_PITCH_CLASSES = [
    'Do', 'Do#', 'Ré', 'Ré#', 'Mi', 'Fa',
    'Fa#', 'Sol', 'Sol#', 'La', 'La#', 'Si'
]

BEAT_VALUES = [
    8,
    7,
    6,
    4,
    3.5,
    3,
    2,
    1.75,
    1.5,
    1,
    0.875,
    0.75,
    0.5,
    0.4375,
    0.375,
    0.25,
    0.21875,
    0.1875,
    0.125,
    0.109375,
    0.09375,
    0.0625,
    float(Fraction(4, 3)),  # Whole-note triplet (3 in 4 beats)
    float(Fraction(2, 3)),  # Quarter-note triplet (3 in 2 beats)
    float(Fraction(1, 3)),  # Eighth-note triplet (3 in 1 beat)
    float(Fraction(1, 6)),  # 16th-note triplet (3 in 0.5 beats)
    float(Fraction(1, 12)),  # 32nd-note triplet (3 in 0.25 beats)
    float(Fraction(2, 3) * 1.5),  # Dotted quarter triplet
    float(Fraction(1, 3) * 1.5),  # Dotted eighth triplet
]


def get_french_pitch_name(midi_number: int) -> str:
    """Convert MIDI number to French pitch name with octave."""
    if not (0 <= midi_number <= 127):
        raise ValueError(f"Invalid MIDI number: {midi_number}. Must be 0-127.")
    octave = (midi_number // 12) - 1
    return f"{FRENCH_PITCH_CLASSES[midi_number % 12]}{octave}"


def adjust_pitch_name(french_name: str, midi_number: int, bend: int) -> str:
    """Adjust pitch name based on bend value."""
    if bend <= 1024:
        return french_name

    # Validate french_name format
    if not french_name or len(french_name) < 2 or not french_name[-1].isdigit():
        raise ValueError(
            f"Invalid french_name '{french_name}' for MIDI number {midi_number}"
        )

    base_name = french_name[:-1]
    octave = french_name[-1]

    if '#' not in base_name:
        return f"{base_name}>{octave}"
    else:
        next_pitch_index = (midi_number % 12 + 1) % 12
        next_pitch = FRENCH_PITCH_CLASSES[next_pitch_index]
        return f"{next_pitch}<{octave}"


def parse_midi_file(midi_path: str) -> pd.DataFrame:
    """Extract quantized notes and rests from MIDI file."""
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        raise ValueError(f"Failed to load MIDI file: {e}") from e

    # Check for valid instruments
    if not midi_data.instruments:
        warnings.warn("MIDI file contains no instruments.", UserWarning)
        return pd.DataFrame(columns=["pitch_name", "pitch_code", "duration"])

    # Tempo handling
    tempo_changes, tempi = midi_data.get_tempo_changes()
    if tempi.any():
        tempo = tempi[0]
        if tempo <= 0:
            raise ValueError(f"Invalid tempo {tempo}. Tempo must be positive.")
    else:
        tempo = 120.0
    beat_duration = 60.0 / tempo

    possible_durations = sorted([v * beat_duration for v in BEAT_VALUES])
    if not possible_durations:
        raise ValueError("No valid durations calculated. Check BEAT_VALUES and tempo.")

    timeline = []
    drum_instruments = 0

    for instrument in midi_data.instruments:
        try:
            if instrument.is_drum:
                drum_instruments += 1
                continue

            for note in instrument.notes:
                # Validate note timing
                if note.end <= note.start:
                    warnings.warn(
                        f"Note end {note.end} <= start {note.start}. Skipping.",
                        UserWarning
                    )
                    continue

                # Calculate original duration
                original_duration = note.end - note.start

                # Get pitch info with validation
                try:
                    french_name = get_french_pitch_name(note.pitch)
                except ValueError as e:
                    warnings.warn(f"Skipping note: {e}", UserWarning)
                    continue

                # Get pitch bend (0 if none found)
                pitch_bend = next(
                    (b.pitch for b in instrument.pitch_bends
                     if abs(b.time - note.start) < 0.01),
                    0
                )

                # Adjust pitch name with validation
                try:
                    final_name = adjust_pitch_name(french_name, note.pitch, pitch_bend)
                except ValueError as e:
                    warnings.warn(f"Skipping note: {e}", UserWarning)
                    continue

                adjusted_pitch = note.pitch + (0.5 if pitch_bend > 1024 else 0)

                # Quantize duration
                index = bisect.bisect_left(possible_durations, original_duration)
                quant_duration = (
                    possible_durations[index]
                    if index < len(possible_durations)
                    else possible_durations[-1]
                )

                timeline.append({
                    'type': 'note',
                    'time': note.start,
                    'end': note.start + quant_duration,
                    'pitch_name': final_name,
                    'pitch_code': adjusted_pitch,
                    'duration': quant_duration
                })

        except Exception as e:
            warnings.warn(f"Error processing instrument: {e}", UserWarning)
            continue

    # Check if all instruments were drums
    if drum_instruments == len(midi_data.instruments):
        warnings.warn("All instruments are drums. No notes extracted.", UserWarning)
        return pd.DataFrame(columns=["pitch_name", "pitch_code", "duration"])

    timeline.sort(key=lambda x: x['time'])
    processed_events = []
    previous_end = 0.0

    for event in timeline:
        # Handle rests between previous end and current event
        if event['time'] > previous_end:
            rest_duration = event['time'] - previous_end
            index = bisect.bisect_left(possible_durations, rest_duration)
            quant_rest = (
                possible_durations[index]
                if index < len(possible_durations)
                else possible_durations[-1]
            )
            if quant_rest > 0:
                processed_events.append({
                    'pitch_name': '*',
                    'pitch_code': 0,
                    'duration': quant_rest
                })

        # Add the quantized note
        processed_events.append({
            'pitch_name': event['pitch_name'],
            'pitch_code': event['pitch_code'],
            'duration': event['duration']
        })
        previous_end = max(previous_end, event['end'])

    return pd.DataFrame(processed_events, columns=["pitch_name", "pitch_code", "duration"])


def pitch_statistics(df):
    # Calculate total metrics
    total_occurrences = len(df)
    total_duration = df['duration'].sum()

    # Group by pitch name and calculate statistics
    grouped = df.groupby('pitch_name', as_index=False).agg(
        occurrence=('duration', 'count'),
        total_duration=('duration', 'sum'),
        mean_duration=('duration', 'mean')
    )

    # Calculate percentages
    grouped['occurrence %'] = (grouped['occurrence'] / total_occurrences) * 100
    grouped['duration %'] = (grouped['total_duration'] / total_duration) * 100

    # Sort by occurrence descending
    grouped = grouped.sort_values('occurrence', ascending=False)

    # Reorder columns to match desired output
    result = grouped[['pitch_name', 'occurrence', 'occurrence %',
                      'total_duration', 'duration %', 'mean_duration']]

    return result


def internal_finals(df):
    # Find all rest positions
    rests = df[df['pitch_name'] == '*'].index

    # Collect preceding notes (internal finals)
    finals = []
    for idx in rests:
        if idx > 0:  # Ensure there's a preceding note
            finals.append(df.iloc[idx - 1]['pitch_name'])

    # Create result dataframe
    if finals:
        result = pd.DataFrame(finals, columns=['pitch_name']) \
            .value_counts().reset_index(name='occurrence')
    else:
        result = pd.DataFrame(columns=['pitch_name', 'occurrence'])

    return result


def melodic_peaks(df):
    # Filter out rests and create working copy
    notes_df = df[df['pitch_name'] != '*'].reset_index(drop=True)

    peaks = []
    troughs = []

    # Need at least 3 notes to identify peaks/troughs
    if len(notes_df) < 3:
        return pd.DataFrame(columns=['pitch_name', 'type', 'count'])

    # Check each middle note for peak/trough status
    for i in range(1, len(notes_df) - 1):
        current = notes_df.at[i, 'pitch_code']
        prev = notes_df.at[i - 1, 'pitch_code']
        next_ = notes_df.at[i + 1, 'pitch_code']

        if current > prev and current > next_:
            peaks.append(notes_df.at[i, 'pitch_name'])
        elif current < prev and current < next_:
            troughs.append(notes_df.at[i, 'pitch_name'])

    # Create result dataframes
    peak_counts = pd.Series(peaks).value_counts().reset_index()
    peak_counts.columns = ['pitch_name', 'ascending_peaks']

    trough_counts = pd.Series(troughs).value_counts().reset_index()
    trough_counts.columns = ['pitch_name', 'descending_peaks']

    # Merge results
    result = pd.merge(peak_counts, trough_counts, on='pitch_name', how='outer').fillna(0)

    # Add total peaks column and sort
    result['total_extremes'] = result['ascending_peaks'] + result['descending_peaks']
    result = result.sort_values('total_extremes', ascending=False).drop('total_extremes', axis=1)

    return result


def pitch_transition_matrix(df):
    # Create list of all transitions
    transitions = []
    for i in range(len(df) - 1):
        current = df.loc[i, 'pitch_name']
        next_pitch = df.loc[i + 1, 'pitch_name']
        transitions.append((current, next_pitch))

    # Create matrix using crosstab
    if transitions:
        from_pitches, to_pitches = zip(*transitions)
        matrix = pd.crosstab(
            pd.Categorical(from_pitches, categories=df['pitch_name'].unique()),
            pd.Categorical(to_pitches, categories=df['pitch_name'].unique()),
            dropna=False
        )
    else:
        matrix = pd.DataFrame()

    # Add row/column labels
    matrix.index.name = "de"
    matrix.columns.name = "vers"

    return matrix


def duration_transition_matrix(df):
    # Get list of durations (including rests)
    durations = df['duration'].tolist()

    # Create transition pairs
    transitions = []
    for i in range(len(durations) - 1):
        current = durations[i]
        next_dur = durations[i + 1]
        transitions.append((current, next_dur))

    # Create matrix using all unique durations as categories
    if transitions:
        unique_durs = sorted(df['duration'].unique())
        from_durs, to_durs = zip(*transitions)

        matrix = pd.crosstab(
            pd.Categorical(from_durs, categories=unique_durs),
            pd.Categorical(to_durs, categories=unique_durs),
            dropna=False
        )
    else:
        matrix = pd.DataFrame()

    # Add labels and sort
    matrix.index.name = "de"
    matrix.columns.name = "vers"
    matrix = matrix.sort_index(axis=0).sort_index(axis=1)

    return matrix


def intervals_statistics(df):
    intervals = []

    # Iterate through consecutive note pairs
    for i in range(len(df) - 1):
        current = df.iloc[i]
        next_note = df.iloc[i + 1]

        # Skip pairs involving rests
        if current['pitch_name'] == '*' or next_note['pitch_name'] == '*':
            continue

        # Calculate interval in semitones (including microtones)
        interval = next_note['pitch_code'] - current['pitch_code']
        intervals.append(round(interval, 2))  # Round to handle floating point precision

    # Create count dataframe
    interval_counts = pd.Series(intervals).value_counts().reset_index()
    interval_counts.columns = ['interval', 'occurrence']

    # Sort by occurrence then by interval
    interval_counts = interval_counts.sort_values(
        ['occurrence', 'interval'],
        ascending=[False, False]
    ).reset_index(drop=True)

    return interval_counts


def interval_transition_matrix(df):
    # First compute intervals between consecutive notes
    intervals = []
    for i in range(len(df) - 1):
        current = df.iloc[i]
        next_note = df.iloc[i + 1]

        # Skip pairs involving rests
        if current['pitch_name'] == '*' or next_note['pitch_name'] == '*':
            continue

        # Calculate interval with microtonal precision
        interval = round(next_note['pitch_code'] - current['pitch_code'], 2)
        intervals.append(interval)

    if len(intervals) < 2:
        return pd.DataFrame()

    # Create transition pairs
    from_intervals = intervals[:-1]
    to_intervals = intervals[1:]

    # Get all unique intervals in sorted order
    unique_intervals = sorted(list(set(intervals)))

    # Create categorical matrix
    matrix = pd.crosstab(
        pd.Categorical(from_intervals, categories=unique_intervals, ordered=True),
        pd.Categorical(to_intervals, categories=unique_intervals, ordered=True),
        dropna=False
    )

    matrix.index.name = "de"
    matrix.columns.name = "vers"

    return matrix

def get_analysis(midi_file):
    results_dict = {
        "pitch_table": "",
        "if_table": "",
        "mp_table": "",
        "pitch_tm": "",
        "duration_tm": "",
        "int_table": "",
        "int_tm": "",
    }

    midi_df = parse_midi_file(midi_file)

    results_dict["pitch_table"] = pitch_statistics(midi_df).to_json(orient="split")
    results_dict["if_table"] = internal_finals(midi_df).to_json(orient="split")
    results_dict["mp_table"] = melodic_peaks(midi_df).to_json(orient="split")
    results_dict["pitch_tm"] = pitch_transition_matrix(midi_df).to_json(orient="split")
    results_dict["duration_tm"] = duration_transition_matrix(midi_df).to_json(orient="split")
    results_dict["int_table"] = intervals_statistics(midi_df).to_json(orient="split")
    results_dict["int_tm"] = interval_transition_matrix(midi_df).to_json(orient="split")

    return results_dict



app = FastAPI(
    title="MIDI Analysis API",
    description="API for analyzing MIDI files and extracting musical metrics",
    version="1.0.0",
)


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://midi-harmony-explorer.lovable.app",
                   "http://tmsd.tn",
                   "https://tmsd.tn",
                   "http://*.tmsd.tn",
                   "https://*.tmsd.tn"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)


@app.post(
    "/",
    response_model=Dict[str, Any],
    summary="Analyze MIDI file",
    description="Process a MIDI file and return musical analysis results",
    responses={
        200: {"description": "Successful analysis"},
        400: {"description": "Invalid file format or processing error"},
        413: {"description": "File too large"},
    }
)
async def analyze_midi(
        file: UploadFile = File(
            ...,
            description="MIDI file to analyze (supported formats: .mid, .midi)",
            max_size=10_000_000,  # 10MB limit
        )
) -> JSONResponse:
    """
    Analyze a MIDI file and extract musical metrics

    Args:
        midi_file: Uploaded MIDI file

    Returns:
        JSON response containing analysis results

    Raises:
        HTTPException: If file processing fails
        :param file: 
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.mid', '.midi')):
            raise ValueError("Invalid file format. Only MIDI files are accepted.")

        # Read and process file
        file_content = await file.read()
        midi_received = BytesIO(file_content)

        # Run analysis
        analysis_result = get_analysis(midi_received)

        return JSONResponse(
            content=analysis_result,
            status_code=200
        )

    except ValueError as ve:
        raise HTTPException(
            status_code=400,
            detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error processing MIDI file: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
