import enums
import numpy as np
import csv
from track_feature_parser import TrackFeatureParser


class DataParser:
    def __init__(self, tf_folder):
        self.track_features = TrackFeatureParser.get_track_features(tf_folder)

    def get_data_from_file(self, file):
        cur_id = None
        session_features = []
        track_features = []
        skips = []
        with open(file) as csvfile:
            reader = csv.DictReader(csvfile)
            session_lines = []
            for row in reader:
                if cur_id is None:
                    cur_id = row['session_id']
                if row['session_id'] == cur_id:
                    session_lines.append(row)
                else:
                    sfs, tfs, skps = self._get_data_from_session(session_lines)
                    session_features.append(sfs)
                    track_features.append(tfs)
                    skips.append(skps)
                    session_lines = [row]
                    cur_id = row['session_id']
        data = {'session_features': session_features, 'track_features': track_features, 'skips': skips}
        return data

    def _get_data_from_session(self, session):
        fraction = int(float(session[0]['session_length']) * 0.5)
        first_half = session[:fraction]
        second_half = session[fraction:]

        skips = self._get_skips(second_half)
        tfs = self._get_track_features(session)
        sfs = self._get_session_metadata(first_half)

        return sfs, tfs, skips

    def _get_skips(self, session):
        out = []
        for sess in session:
            out.append([DataParser.get_numeric_bool_from_string(sess['skip_2'])])
        return np.array(out, np.float32)

    def _get_session_metadata(self, session):
        out = []
        for line in session:
            row = line
            del row['session_id']
            del row['track_id_clean']
            del row['date']
            row['skip_1'] = DataParser.get_numeric_bool_from_string(row['skip_1'])
            row['skip_2'] = DataParser.get_numeric_bool_from_string(row['skip_2'])
            row['skip_3'] = DataParser.get_numeric_bool_from_string(row['skip_3'])
            row['not_skipped'] = DataParser.get_numeric_bool_from_string(row['not_skipped'])
            row['hist_user_behavior_is_shuffle'] = DataParser.get_numeric_bool_from_string(row['hist_user_behavior_is_shuffle'])
            row['premium'] = DataParser.get_numeric_bool_from_string(row['premium'])
            row['context_type'] = DataParser.get_spotify_context_type_from_string(row['context_type'])
            row['hist_user_behavior_reason_start'] = DataParser.get_reason_track_start_from_string(row['hist_user_behavior_reason_start'])
            row['hist_user_behavior_reason_end'] = DataParser.get_reason_track_end_from_string(row['hist_user_behavior_reason_end'])
            ls = list(row.values())
            out.append(ls)
        return np.array(out, np.float32)

    def _get_track_features(self, session):
        out = []
        for line in session:
            out.append(self.track_features[line['track_id_clean']])
        return np.array(out, np.float32)

    # static methods
    @staticmethod
    # count, skip1, skip2, skip3, not skipped
    def append_track_data(row, tracks):
        s1 = DataParser.get_numeric_bool_from_string(row['skip_1'])
        s2 = DataParser.get_numeric_bool_from_string(row['skip_2'])
        s3 = DataParser.get_numeric_bool_from_string(row['skip_3'])
        ns = DataParser.get_numeric_bool_from_string(row['not_skipped'])
        track_data = np.array([1, s1, s2, s3, ns], np.float32)
        if row['track_id_clean'] not in tracks:
            tracks[row['track_id_clean']] = track_data
        else:
            tracks[row['track_id_clean']] += track_data

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
