import tracklog_enums as enums
import numpy as np
import csv


class LogParser:
    @staticmethod
    def get_input_data_from_csv_file(file):
        sessions = {}
        with open(file) as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            # skip the headers
            next(reader, None)
            for row in reader:
                if row[0] in sessions:
                    sessions[row[0]].append(LogParser.process_input_csv_line(row[1:]))
                else:
                    sessions[row[0]] = [LogParser.process_input_csv_line(row[1:])]
        for session_id in sessions:
            sessions[session_id] = np.array(sessions[session_id], dtype=float)
        return sessions

    @staticmethod
    def get_tracks_statistics(file, tracks):
        with open(file) as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            # skip the headers
            next(reader, None)
            for row in reader:
                LogParser.append_track_data(row, tracks)

    @staticmethod
    # count, skip1, skip2, skip3, not skipped
    def append_track_data(row, tracks):
        s1 = LogParser.get_numeric_bool_from_string(row[4])
        s2 = LogParser.get_numeric_bool_from_string(row[5])
        s3 = LogParser.get_numeric_bool_from_string(row[6])
        ns = LogParser.get_numeric_bool_from_string(row[7])
        track_data = np.array([1, s1, s2, s3, ns], np.float32)
        if row[3] not in tracks:
            tracks[row[3]] = track_data
        else:
            tracks[row[3]] += track_data

    @staticmethod
    def get_raw_data_from_csv_file(file):
        return np.genfromtxt(file, delimiter=',', skip_header=1, dtype=str)

    @staticmethod
    def get_output_data_from_csv_file(file):
        sessions = {}
        with open(file) as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            # skip the headers
            next(reader, None)
            for row in reader:
                if row[0] in sessions:
                    sessions[row[0]].append(LogParser.process_input_csv_line(row[1:]))
                else:
                    sessions[row[0]] = [LogParser.process_input_csv_line(row[1:])]
        for session_id in sessions:
            sessions[session_id] = np.array(sessions[session_id], dtype=float)
        return sessions

    @staticmethod
    def process_output_csv_line(line):
        line = [LogParser.get_numeric_bool_from_string(l) for l in line]
        return line

    @staticmethod
    def process_input_csv_line(line):
        del line[15]
        del line[1:7]
        line[7] = LogParser.get_numeric_bool_from_string(line[7])
        line[9] = LogParser.get_numeric_bool_from_string(line[9])
        line[10] = LogParser.get_spotify_context_type_from_string(line[10])
        line[11] = LogParser.get_reason_track_start_from_string(line[11])
        line[12] = LogParser.get_reason_track_end_from_string(line[12])
        return line

    @staticmethod
    def get_numeric_bool_from_string(string):
        if string == "false" or string == "0":
            return 0.
        elif string == "true" or string == "1":
            return 1.

    @staticmethod
    def get_spotify_context_type_from_string(string):
        if string == "editorial_playlist":
            return enums.SpotifyContextType.EDITORIAL_PLAYLIST
        elif string == "catalog":
            return enums.SpotifyContextType.CATALOG
        elif string == "radio":
            return enums.SpotifyContextType.RADIO
        elif string == "charts":
            return enums.SpotifyContextType.CHARTS
        elif string == "user_collection":
            return enums.SpotifyContextType.USER_COLLECTION
        elif string == "personalized_playlist":
            return enums.SpotifyContextType.PERSONALIZED_PLAYLIST
        else:
            return enums.SpotifyContextType.UNDEFINED

    @staticmethod
    def get_reason_track_start_from_string(string):
        if string == "trackdone":
            return enums.ReasonTrackStart.TRACK_DONE
        elif string == "fwdbtn":
            return enums.ReasonTrackStart.FORWARD_BUTTON
        elif string == "backbtn":
            return enums.ReasonTrackStart.BACKWARD_BUTTON
        elif string == "clickrow":
            return enums.ReasonTrackStart.CLICK_ROW
        elif string == "playbtn":
            return enums.ReasonTrackStart.PLAY_BUTTON
        elif string == "trackerror":
            return enums.ReasonTrackStart.TRACK_ERROR
        elif string == "appload":
            return enums.ReasonTrackStart.APP_LOAD
        elif string == "popup":
            return enums.ReasonTrackStart.POP_UP
        elif string == "uriopen":
            return enums.ReasonTrackStart.URI_OPEN
        elif string == "clickrow":
            return enums.ReasonTrackStart.CLICK_ROW
        elif string == "clickside":
            return enums.ReasonTrackStart.CLICK_SIDE
        elif string == "remote":
            return enums.ReasonTrackStart.REMOTE
        else:
            return enums.ReasonTrackStart.UNDEFINED

    @staticmethod
    def get_reason_track_end_from_string(string):
        if string == "fwdbtn":
            return enums.ReasonTrackEnd.FORWARD_BUTTON
        elif string == "backbtn":
            return enums.ReasonTrackEnd.BACKWARD_BUTTON
        elif string == "trackdone":
            return enums.ReasonTrackEnd.TRACK_DONE
        elif string == "endplay":
            return enums.ReasonTrackEnd.END_PLAY
        elif string == "logout":
            return enums.ReasonTrackEnd.LOGOUT
        elif string == "trackerror":
            return enums.ReasonTrackEnd.TRACK_ERROR
        elif string == "remote":
            return enums.ReasonTrackEnd.REMOTE
        elif string == "popup":
            return enums.ReasonTrackEnd.POP_UP
        elif string == "appload":
            return enums.ReasonTrackEnd.APP_LOAD
        elif string == "uriopen":
            return enums.ReasonTrackEnd.URI_OPEN
        elif string == "clickrow":
            return enums.ReasonTrackEnd.CLICK_ROW
        elif string == "clickside":
            return enums.ReasonTrackEnd.CLICK_SIDE
        else:
            return enums.ReasonTrackEnd.UNDEFINED
