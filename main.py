from io import BytesIO
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Any
import uvicorn
from pandas import Series, concat, DataFrame
from pretty_midi import PrettyMIDI
from itertools import islice

# Dictionary mapping duration names in different languages
DURATION_NAMES = {
    'EN': ["whole(..)",
           "whole(.)",
           "whole",
           "half(..)",
           "half(.)",
           "half",
           "quarter(..)",
           "quarter(.)",
           "quarter",
           "eighth(..)",
           "eighth(.)",
           "eighth",
           "sixteenth(..)",
           "sixteenth(.)",
           "sixteenth",
           "thirty-second(..)",
           "thirty-second(.)",
           "thirty-second",
           "sixty-fourth(..)",
           "sixty-fourth(.)",
           "sixty-fourth"],

    'FR': ["ronde(..)",
           "ronde(.)",
           "ronde",
           "blanche(..)",
           "blanche(.)",
           "blanche",
           "noire(..)",
           "noire(.)",
           "noire",
           "croche(..)",
           "croche(.)",
           "croche",
           "double-croche(..)",
           "double-croche(.)",
           "double-croche",
           "triple-croche(..)",
           "triple-croche(.)",
           "triple-croche",
           "quadruple-croche(..)",
           "quadruple-croche(.)",
           "quadruple-croche"],

    'AR': ["مستديرة(..)",
           "مستديرة(.)",
           "مستديرة",
           "بيضاء(..)",
           "بيضاء(.)",
           "بيضاء",
           "سوداء(..)",
           "سوداء(.)",
           "سوداء",
           "مشالة(..)",
           "مشالة(.)",
           "مشالة",
           "2 شيلات(..)",
           "2 شيلات(.)",
           "2 شيلات",
           "3 شيلات(..)",
           "3 شيلات(.)",
           "3 شيلات",
           "4 شيلات(..)",
           "4 شيلات(.)",
           "4 شيلات"]
}

# Dictionary mapping pitch names in different languages
PITCH_NAMES = {
    'EN': ["C", "C>", "C#", "D<", "D", "D>", "D#", "E<", "E", "E>", "F", "F>",
           "F#", "G<", "G", "G>", "G#", "A<", "A", "A>", "A#", "B<", "B", "B>"],
    'FR': ["do", "do>", "do#", "re<", "re", "re>", "re#", "mi<", "mi", "mi>", "fa", "fa>",
           "fa#", "sol<", "sol", "sol>", "sol#", "la<", "la", "la>", "la#", "si<", "si", "si>"],
    'AR': {
        "G3": "يك-كاه",
        "G>3": "تيك يك-كاه",
        "G#3": "قرار حصار",
        "A<3": "نيم عشيران",
        "A3": "عشيران",
        "A>3": "نيم عجم عشيران",
        "A#3": "نيم عراق",
        "B<3": "عراق",
        "B3": "كواشت",
        "B>3": "تيك كواشت",
        "C4": "راست",
        "C>4": "تيك راست",
        "C#4": "زير كوله",
        "D<4": "تيك زير كوله",
        "D4": "دو–كاه",
        "D>4": "تيك دو-كاه",
        "D#4": "كردى حجاز",
        "E<4": "سه–كاه",
        "E4": "بوسلك",
        "E>4": "تيك بوسلك",
        "F4": "تشهار–كاه",
        "F>4": "تيك تشهار–كاه",
        "F#4": "حجاز",
        "G<4": "عزال",
        "G4": "نوى",
        "G>4": "تيك نوى",
        "G#4": "حصار",
        "A<4": "نيم حسينى",
        "A4": "حسينى",
        "A>4": "نيم عجم",
        "A#4": "نيم اوج",
        "B<4": "اوج",
        "B4": "ماهور",
        "B>4": "تيك ماهور",
        "C5": "كردان",
        "C>5": "تيك كردان",
        "C#5": "جواب زير كوله",
        "D<5": "جواب تيك زير كوله",
        "D5": "محير",
        "D>5": "تيك محير",
        "D#5": "سنبله حجاز",
        "E<5": "بزرك",
        "E5": "جواب بوسلك",
        "E>5": "جواب تيك بوسلك",
        "F5": "ماهوران",
        "F>5": "تيك ماهوران",
        "F#5": "جواب حجاز",
        "G<5": "جواب عزال",
        "G5": "سهم"
    }
}

# Dictionary mapping columns names in different languages
COLUMNS_LANGUAGE = {
    "EN": {
        "analyze_pitches": ["Pitches", "Occurrence", "%_Occ", "Duration", "%_Dur", "Mean Duration"],
        "internal_final": ["Internal Finals", "Occurrence"],
        "melodic_peaks": ["Pitch", "Upper Peak", "Lower Peak"],
        "pitch_transition_matrix": ["Pitch 1", "Pitch 2"],
        "duration_transition_matrix": ["Duration 1", "Duration 2"],
        "interval_count": ["Interval", "Occurrence"],
        "interval_transition_matrix": ["Interval 1", "Interval 2"]
    },
    "FR": {
        "analyze_pitches": ["Notes", "Occurence", "%_Occ", "Durée", "%_Dur", "Durée Moyenne"],
        "internal_final": ["Finale Interne", "Occurence"],
        "melodic_peaks": ["Note", "Pic Supérieur", "Pic Inférieur"],
        "pitch_transition_matrix": ["Note 1", "Note 2"],
        "duration_transition_matrix": ["Durée 1", "Durée 2"],
        "interval_count": ["Intervalle", "Occurence"],
        "interval_transition_matrix": ["Intervalle 1", "Intervalle 2"]
    },
    "AR": {
        "analyze_pitches": ["الدرجة", "التواتر", "% التواتر", "المدّة", "% المدّة", "متوسط المدّة"],
        "internal_final": ["ارتكاز داخلي", "التواتر"],
        "melodic_peaks": ["الدرجة", "القمة العليا", "الذروة السفلية"],
        "pitch_transition_matrix": ["الدرجة 1", "الدرجة 2"],
        "duration_transition_matrix": ["المدّة 1", "المدّة 2"],
        "interval_count": ["البعد", "التواتر"],
        "interval_transition_matrix": ["البعد 1", "البعد 2"]
    }
}

def window(seq, n=2):
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

class TMSDMidiParser:
    """
    class to read a midi file from a path
    and convert it content to a dataframe
    """
    @staticmethod
    def quantize_duration(duration, durations_list):
        """
            Quantize a duration value to the closest value from the durations list.

            Args:
                duration (float): The duration value to be quantized.
                durations_list (list): List of durations to compare with.

            Returns:
                float: The quantized duration value.
            """

        for i, dur in enumerate(durations_list):
            if duration > dur:
                return durations_list[i - 1]
            if duration == dur:
                return durations_list[i]

    @staticmethod
    def get_duration_list(tempo):
        """
            Generates a list of durations based on the given tempo.

            Args:
                tempo (int): The tempo value.

            Returns:
                list: List of durations.
            """

        qnd = 60_000_000 / (tempo * 1000000)  # quarter_note_duration

        durations = [
            7 * qnd,
            6 * qnd,
            4 * qnd,
            3.5 * qnd,
            3 * qnd,
            2 * qnd,
            1.75 * qnd,
            1.5 * qnd,
            qnd,
            0.875 * qnd,
            0.75 * qnd,
            0.5 * qnd,
            0.4375 * qnd,
            0.375 * qnd,
            0.25 * qnd,
            0.21875 * qnd,
            0.1875 * qnd,
            0.125 * qnd,
            0.109375 * qnd,
            0.09375 * qnd,
            0.0625 * qnd
        ]
        return durations

    @staticmethod
    def get_pitch_name(pitch_number, pitch_bend, language):
        """
            Retrieves the pitch name based on the pitch number, pitch bend, and language.

            Args:
                pitch_number (int): The MIDI pitch number.
                pitch_bend (int): The pitch bend value.
                language (str): The language code.

            Returns:
                str: The pitch name.
            """

        note_number = int(round(pitch_number))
        en_semis = PITCH_NAMES['EN']
        if pitch_bend != 0:
            en_name = en_semis[((note_number % 12) * 2) + 1] + str(note_number // 12 - 1)
        else:
            en_name = en_semis[(note_number % 12) * 2] + str(note_number // 12 - 1)
        print(language)
        match language:
            case "EN":
                return en_name
            case "FR":
                return PITCH_NAMES['FR'][en_semis.index(en_name[:-1])] + en_name[-1]
            case "AR":
                if pitch_number < 55 or pitch_number > 79:
                    return en_name
                else:
                    return PITCH_NAMES['AR'][en_name]

    @staticmethod
    def get_duration_name(q_dur, durations_list, language):
        """
            Retrieves the duration name based on the quantized duration, durations list, and language.

            Args:
                q_dur (float): The quantized duration value.
                durations_list (list): List of durations.
                language (str): The language code.

            Returns:
                str: The duration name.
            """

        duration_index = durations_list.index(q_dur)
        match language:
            case "EN":
                return DURATION_NAMES["EN"][duration_index]
            case "FR":
                return DURATION_NAMES["FR"][duration_index]
            case "AR":
                return DURATION_NAMES["AR"][duration_index]

    def parse_midi_file(self, midi_file_path, language):
        """
            Parses a MIDI file and returns the data as a DataFrame.

            Args:
                midi_file_path (str): The path to the MIDI file.
                language (str): The language code.

            Returns:
                pandas.DataFrame: The parsed MIDI data as a DataFrame.
            """

        midi_data = PrettyMIDI(midi_file_path)
        data = []

        instrument = midi_data.instruments[0]
        tempo = round(midi_data.get_tempo_changes()[1][0])
        durations_list = self.get_duration_list(tempo)
        smallest_duration = durations_list[-1]
        prev_end_time = 0.0

        for note in instrument.notes:
            start = note.start
            end = note.end
            pitch_bend = 0
            dur = round(end - start, 6)
            pitch_code = note.pitch

            q_dur = self.quantize_duration(dur, durations_list)
            duration_name = self.get_duration_name(q_dur, durations_list, language)

            end = start + q_dur

            for bend in instrument.pitch_bends:
                if bend.time == note.start and bend.pitch > 10:
                    pitch_bend = 2048
                    pitch_code += 0.5

            if start - prev_end_time >= smallest_duration:
                rest_duration = self.quantize_duration(start - prev_end_time, durations_list)
                rest_dur_name = self.get_duration_name(rest_duration, durations_list, language)
                data.append([prev_end_time, start, '*', rest_duration, '*', '*', rest_dur_name])

            pitch = self.get_pitch_name(note.pitch, pitch_bend, language)

            data.append([start, end, pitch, q_dur, pitch_bend, pitch_code, duration_name])
            prev_end_time = end

        midi_list = sorted(data, key=lambda x: x[0])

        data_frame = DataFrame(midi_list, columns=['Start', 'End', 'Pitch', 'Duration', 'Bend', 'Code', 'Dur_Name'])

        return data_frame

class Analyzer:
    """
    class to read a dataframe containing converted midi file
    and compute musical information and returns a dictionary
    """
    def __init__(self, midi_file, language):

        # Read the MIDI file into a DataFrame
        parser_instance = TMSDMidiParser()
        self.midi_df = parser_instance.parse_midi_file(midi_file, language)
        self.language = language

    def analyze_pitches(self):
        """
        Analyze the pitch information in a MIDI file.

        Args:
            midi_df (pandas.DataFrame): A pandas DataFrame with the MIDI data, containing at least the columns "Pitch"
        and "Duration".

        Returns:
            str: A JSON string representing a pandas DataFrame with the analysis results, containing columns for the
        pitches, their occurrences, their percentage of total occurrences, their durations, their percentage of total
        duration, and their average duration.

        """
        midi_df = self.midi_df

        # create a new DataFrame to store the analysis results
        pitches = DataFrame(
            columns=["Pitches", "Occurrence", "%_Occ", "Duration", "%_Dur", "Mean Duration"]
        )

        # count the number of occurrences for each note and store them in the new DataFrame
        val_count_haut = midi_df.Pitch.value_counts()
        pitches["Pitches"] = val_count_haut.index
        pitches["Occurrence"] = val_count_haut.values

        # calculate the percentage of total occurrences for each note and store it in the new DataFrame
        pitches["%_Occ"] = (
                100 * midi_df.Pitch.value_counts(normalize=True).values
        )

        # calculate the total duration for each note and store it in the new DataFrame
        duree = midi_df.Duration.groupby(midi_df.Pitch).sum()
        for i, v in duree.items():
            pitches.loc[pitches.index[pitches["Pitches"] == i], ["Duration"]] = v

        # calculate the percentage of total duration for each note and store it in the new DataFrame
        pitches["%_Dur"] = 100 * (pitches["Duration"] / pitches["Duration"].sum())

        # calculate the average duration for each note and store it in the new DataFrame
        pitches["Mean Duration"] = pitches["Duration"] / pitches["Occurrence"]

        # return the analysis results as a JSON string in the format of a pandas DataFrame
        return pitches.to_json(orient="split")

    def internal_final(self):
        """
        This function takes in a dataframe of pitch data for a midi file, and returns a JSON string of the count
        of internal final pitches in the piece of music.

        Args:
            pitches_df (pandas.DataFrame): A dataframe of pitch data for a midi file.

        Returns:
            A JSON string of the count of internal final pitches in the piece of music.

        """
        midi_df = self.midi_df
        fin_list = []

        for i, pitches in midi_df.iloc[1:, :].iterrows():
            if pitches["Pitch"] == "*":
                if midi_df["Pitch"].iat[i - 1] != "*":
                    fin_list.append(midi_df["Pitch"].iat[i - 1])

        fin_int = DataFrame(Series(fin_list).value_counts())
        fin_int = fin_int.reset_index()
        fin_int = fin_int.rename(columns={"index": "Internal Finals", 0: "Occurrence"})

        return fin_int.to_json(orient="split")

    def melodic_peaks(self):
        """
        This function takes a MIDI file in DataFrame format as input and identifies the melodic peaks of the song.

        Args:
            midi_df (pandas.DataFrame): a DataFrame containing MIDI data with columns 'Pitch', 'Velocity', 'Time' and
        'Duration'.

        Returns:
            pandas.DataFrame: a DataFrame containing the melodic peaks of the song with columns 'Pitch', 'Upper Peak'
        and 'Lower Peak'.

        """
        midi_df = self.midi_df

        # Select only pitches (rows) that are not marked with a *
        song = midi_df[midi_df.Pitch != "*"]

        # Reset the index of the DataFrame after removing the rows marked with *
        song = song.reset_index()

        # Drop the index column
        song = song.drop(["index"], axis=1)

        # Create an empty DataFrame to hold the melodic peaks
        melo_peak = DataFrame(columns=["Pitch", "Upper Peak", "Lower Peak"])

        # Loop through each row of the song DataFrame to identify the melodic peaks
        for i, pitches in song.iloc[1:-1, :].iterrows():
            # Calculate the superior peak (note is higher than the surrounding pitches)
            if (song["Code"].iat[i] > song["Code"].iat[i - 1]) and (
                    song["Code"].iat[i] > song["Code"].iat[i + 1]
            ):
                # If the note is not already in the DataFrame, add it with one superior peak
                if song["Pitch"].iat[i] not in melo_peak["Pitch"].values:
                    new_row = DataFrame(
                        {
                            "Pitch": song["Pitch"].iat[i],
                            "Upper Peak": [1],
                            "Lower Peak": [0],
                        }
                    )
                    melo_peak = concat([melo_peak, new_row], ignore_index=True)

                # If the note is already in the DataFrame, increment its superior peak count
                else:
                    index = melo_peak.index[
                        melo_peak["Pitch"].values == song["Pitch"].iat[i]
                        ][0]
                    melo_peak.loc[index, ["Upper Peak"]] = [
                        melo_peak["Upper Peak"].iat[index] + 1
                    ]

            # Calculate the inferior peak (note is lower than the surrounding pitches)
            elif (song["Code"].iat[i] < song["Code"].iat[i - 1]) and (
                    song["Code"].iat[i] < song["Code"].iat[i + 1]
            ):
                # If the note is not already in the DataFrame, add it with one inferior peak
                if song["Pitch"].iat[i] not in melo_peak["Pitch"].values:
                    new_row = DataFrame(
                        {
                            "Pitch": song["Pitch"].iat[i],
                            "Upper Peak": [0],
                            "Lower Peak": [1],
                        }
                    )

                    melo_peak = concat([melo_peak, new_row], ignore_index=True)

                # If the note is already in the DataFrame, increment its inferior peak count
                else:
                    index = melo_peak.index[
                        melo_peak["Pitch"].values == song["Pitch"].iat[i]
                        ][0]
                    melo_peak.loc[index, ["Lower Peak"]] = [
                        melo_peak["Lower Peak"].iat[index] + 1
                    ]

        # Return the DataFrame containing the melodic peaks of the song
        return melo_peak.to_json(orient="split")

    def transition_matrix(self):
        """
        Calculates the transition matrices for pitches and durations of a MIDI file.

        Args:
            midi_df (pandas.DataFrame): A pandas DataFrame containing MIDI data, with columns for 'Pitch' and 'Duration'.

        Returns:
            Tuple: A tuple containing two pandas DataFrames in JSON format representing the transition matrices for pitches
            and durations respectively.

        """
        midi_df = self.midi_df

        # Get the list of pitches and durations from the MIDI file
        pitches = midi_df.Pitch.tolist()
        durations = midi_df.Dur_Name.tolist()

        # Calculate the transition matrix for pitches
        # Convert the pitches list into a DataFrame with sliding windows of two consecutive pitches
        pairs = DataFrame(window(pitches), columns=["Pitch 1", "Pitch 2"])
        # Count the occurrences of each pair of consecutive pitches and group by the first note in the pair
        counts = pairs.groupby("Pitch 1")["Pitch 2"].value_counts()
        # Calculate the probabilities of transitioning from each note to each other note
        probs = (round(counts / counts.sum(), 3) * 100).unstack()
        # Convert the resulting DataFrame into a matrix where rows
        # represent starting pitches and columns represent ending
        # pitches
        DF_probs = DataFrame(probs)
        pitches_tm_df = DF_probs.fillna(0)

        # Calculate the transition matrix for durations
        # Convert the durations list into a DataFrame with sliding windows of two consecutive durations
        pairs = DataFrame(window(durations), columns=["Duration 1", "Duration 2"])
        # Count the occurrences of each pair of consecutive durations and group by the first duration in the pair
        counts = pairs.groupby("Duration 1")["Duration 2"].value_counts()
        # Calculate the probabilities of transitioning from each duration to each other duration
        probs = (round(counts / counts.sum(), 3) * 100).unstack()
        # Convert the resulting DataFrame into a matrix where rows represent starting durations and columns represent
        # ending durations
        DF_probs = DataFrame(probs)
        duration_tm_df = DF_probs.fillna(0)

        # Return the transition matrices as a tuple of pandas DataFrames in JSON format
        return pitches_tm_df.to_json(orient="split"), duration_tm_df.to_json(
            orient="split"
        )

    def intervals(self):
        """
        This function computes the intervals between consecutive pitches in the MIDI file and returns a dataframe
        with the count of each interval, as well as a transition matrix of interval probabilities.

        Args:
            midi_df (pandas.DataFrame): Dataframe containing the MIDI data, with columns for Pitch, Code, and Duration.

        Returns:
            tuple: A tuple containing two JSON-formatted strings representing the interval count dataframe
            and the interval transition matrix dataframe, respectively.

        """
        midi_df = self.midi_df

        # Get the list of codes from the MIDI dataframe
        codes = midi_df[midi_df.Code != "*"].Code.tolist()

        # Compute the intervals between consecutive pitches
        int_list = []
        for i in range(len(codes) - 1):
            int_list.append(codes[i + 1]-codes[i])

        # Count the occurrences of each interval
        intervals = DataFrame(int_list).value_counts()
        intervals_df = DataFrame(intervals)

        # Compute the transition matrix of interval probabilities
        pairs = DataFrame(window(int_list), columns=["Interval 1", "Interval 2"])
        counts = pairs.groupby("Interval 1")["Interval 2"].value_counts()
        probs = (round(counts / counts.sum(), 3) * 100).unstack()
        DF_probs = DataFrame(probs)
        intervals_tm_df = DF_probs.fillna(0)

        # Return the interval count and transition matrix dataframes as JSON-formatted strings
        return intervals_df.to_json(orient="split"), intervals_tm_df.to_json(
            orient="split"
        )

    def analyzer(self):
        """
        Analyzes the MIDI file and returns a dictionary of analysis results.

        Returns:
            a dictionary containing the following keys:
            "pitch_table": a table of pitch occurrences in the MIDI file,
            "if_table": a table of internal and final intervals in the MIDI file,
            "mp_table": a table of melodic peaks in the MIDI file,
            "pitch_tm": a transition matrix of pitches in the MIDI file,
            "duration_tm": a transition matrix of note durations in the MIDI file,
            "int_table": a table of all intervals in the MIDI file,
            "int_tm": a transition matrix of intervals in the MIDI file
        """
        results_dict = {
            "pitch_table": "",
            "if_table": "",
            "mp_table": "",
            "pitch_tm": "",
            "duration_tm": "",
            "int_table": "",
            "int_tm": "",
        }

        # Generate pitch occurrence table
        pitch_table = self.analyze_pitches()

        # Generate internal and final interval table
        if_table = self.internal_final()

        # Generate melodic peaks table
        mp_table = self.melodic_peaks()

        # Generate transition matrices for pitches and note durations
        pitch_tm, duration_tm = self.transition_matrix()

        # Generate table and transition matrix for all intervals
        int_table, int_tm = self.intervals()

        # Store all analysis results in the dictionary
        results_dict["pitch_table"] = pitch_table
        results_dict["if_table"] = if_table
        results_dict["mp_table"] = mp_table
        results_dict["pitch_tm"] = pitch_tm
        results_dict["duration_tm"] = duration_tm
        results_dict["int_table"] = int_table
        results_dict["int_tm"] = int_tm

        return results_dict


app = FastAPI(
    title="MIDI Analysis API",
    description="API for analyzing MIDI files and extracting musical metrics",
    version="1.0.0",
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
        midi_file: UploadFile = File(
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
    """
    try:
        # Validate file type
        if not midi_file.filename.lower().endswith(('.mid', '.midi')):
            raise ValueError("Invalid file format. Only MIDI files are accepted.")

        # Read and process file
        file_content = await midi_file.read()
        midi_received = BytesIO(file_content)

        # Run analysis
        analyzer_instance = Analyzer(midi_received, 'EN')
        analysis_result = analyzer_instance.analyzer()

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