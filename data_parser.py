import enums
import numpy as np
import csv
from track_feature_parser import TrackFeatureParser
from dataset_description import *


class DataParser:
    def __init__(self, tf_folder):
        self.track_features = TrackFeatureParser.get_track_features(tf_folder)

    def get_data_from_file(self, file):
        cur_id = None
        session_features_first_half = []
        session_features_second_half = []
        track_features_first_half = []
        track_features_second_half = []
        skips = []
        with open(file) as csvfile:
            reader = csv.DictReader(csvfile)
            session_lines = []
            for row in reader:
                if cur_id is None:
                    cur_id = row[SessionFeaturesFields.SESSION_ID]
                if row[SessionFeaturesFields.SESSION_ID] == cur_id:
                    session_lines.append(row)
                else:
                    sfs_first_half, sfs_second_half, tf_first_half, tf_second_half, s_skips = self._get_data_from_session(session_lines)
                    session_features_first_half.append(sfs_first_half)
                    session_features_second_half.append(sfs_second_half)
                    track_features_first_half.append(tf_first_half)
                    track_features_second_half.append(tf_second_half)
                    skips.append(s_skips)
                    session_lines = [row]
                    cur_id = row[SessionFeaturesFields.SESSION_ID]
        data = {DatasetDescription.SF_FIRST_HALF: session_features_first_half,
                DatasetDescription.SF_SECOND_HALF: session_features_second_half,
                DatasetDescription.TF_FIRST_HALF: track_features_first_half,
                DatasetDescription.TF_SECOND_HALF: track_features_second_half,
                DatasetDescription.SKIPS: skips}
        return data

    def _get_data_from_session(self, session):
        fraction = int(float(session[0][SessionFeaturesFields.SESSION_LENGTH]) * 0.5)
        first_half = session[:fraction]
        second_half = session[fraction:]

        skips = self._get_skips(second_half)
        tfs_first_half = self._get_track_features(first_half)
        tfs_second_half = self._get_track_features(second_half)
        sfs_first_half = self._get_session_metadata(first_half)
        sfs_second_half = self._get_session_metadata(second_half)

        return sfs_first_half, sfs_second_half, tfs_first_half, tfs_second_half, skips

    def _get_skips(self, session):
        out = []
        for sess in session:
            out.append([DataParser.get_numeric_bool_from_string(sess[SessionFeaturesFields.SKIP_2])])
        return np.array(out, np.float32)

    def _get_session_metadata(self, session):
        out = []
        for line in session:
            row = line
            del row[SessionFeaturesFields.SESSION_ID]
            del row[SessionFeaturesFields.TRACK_ID_CLEAN]
            del row[SessionFeaturesFields.DATE]
            row[SessionFeaturesFields.SKIP_1] = DataParser.get_numeric_bool_from_string(row[SessionFeaturesFields.SKIP_1])
            row[SessionFeaturesFields.SKIP_2] = DataParser.get_numeric_bool_from_string(row[SessionFeaturesFields.SKIP_2])
            row[SessionFeaturesFields.SKIP_3] = DataParser.get_numeric_bool_from_string(row[SessionFeaturesFields.SKIP_3])
            row[SessionFeaturesFields.NOT_SKIPPED] = DataParser.get_numeric_bool_from_string(row[SessionFeaturesFields.NOT_SKIPPED])
            row[SessionFeaturesFields.HIST_USER_BEHAVIOR_IS_SHUFFLE] = DataParser.get_numeric_bool_from_string(row[SessionFeaturesFields.HIST_USER_BEHAVIOR_IS_SHUFFLE])
            row[SessionFeaturesFields.PREMIUM] = DataParser.get_numeric_bool_from_string(row[SessionFeaturesFields.PREMIUM])
            row[SessionFeaturesFields.CONTEXT_TYPE] = DataParser.get_spotify_context_type_from_string(row[SessionFeaturesFields.CONTEXT_TYPE])
            row[SessionFeaturesFields.HIST_USER_BEHAVIOR_REASON_START] = DataParser.get_reason_track_start_from_string(row[SessionFeaturesFields.HIST_USER_BEHAVIOR_REASON_START])
            row[SessionFeaturesFields.HIST_USER_BEHAVIOR_REASON_END] = DataParser.get_reason_track_end_from_string(row[SessionFeaturesFields.HIST_USER_BEHAVIOR_REASON_END])
            ls = list(row.values())
            out.append(ls)
        return np.array(out, np.float32)

    def _get_track_features(self, session):
        out = []
        for line in session:
            out.append(self.track_features[line[SessionFeaturesFields.TRACK_ID_CLEAN]])
        return np.array(out, np.float32)

    # static methods
    @staticmethod
    # count, skip1, skip2, skip3, not skipped
    def append_track_data(row, tracks):
        s1 = DataParser.get_numeric_bool_from_string(row[SessionFeaturesFields.SKIP_1])
        s2 = DataParser.get_numeric_bool_from_string(row[SessionFeaturesFields.SKIP_2])
        s3 = DataParser.get_numeric_bool_from_string(row[SessionFeaturesFields.SKIP_3])
        ns = DataParser.get_numeric_bool_from_string(row[SessionFeaturesFields.NOT_SKIPPED])
        track_data = np.array([1, s1, s2, s3, ns], np.float32)
        if row[SessionFeaturesFields.TRACK_ID_CLEAN] not in tracks:
            tracks[row[SessionFeaturesFields.TRACK_ID_CLEAN]] = track_data
        else:
            tracks[row[SessionFeaturesFields.TRACK_ID_CLEAN]] += track_data

    @staticmethod
    def get_numeric_bool_from_string(string):
        if string == "false" or string == "0":
            return 0.
        elif string == "true" or string == "1":
            return 1.

    @staticmethod
    def get_spotify_context_type_from_string(string):
        if string == SpotifyContextType.EDITORIAL_PLAYLIST:
            return enums.SpotifyContextType.EDITORIAL_PLAYLIST
        elif string == SpotifyContextType.CATALOG:
            return enums.SpotifyContextType.CATALOG
        elif string == SpotifyContextType.RADIO:
            return enums.SpotifyContextType.RADIO
        elif string == SpotifyContextType.CHARTS:
            return enums.SpotifyContextType.CHARTS
        elif string == SpotifyContextType.USER_COLLECTION:
            return enums.SpotifyContextType.USER_COLLECTION
        elif string == SpotifyContextType.PERSONALIZED_PLAYLIST:
            return enums.SpotifyContextType.PERSONALIZED_PLAYLIST
        else:
            return enums.SpotifyContextType.UNDEFINED

    @staticmethod
    def get_reason_track_start_from_string(string):
        if string == ReasonTrackChange.TRACK_DONE:
            return enums.ReasonTrackStart.TRACK_DONE
        elif string == ReasonTrackChange.FORWARD_BUTTON:
            return enums.ReasonTrackStart.FORWARD_BUTTON
        elif string == ReasonTrackChange.BACK_BUTTON:
            return enums.ReasonTrackStart.BACKWARD_BUTTON
        elif string == ReasonTrackChange.CLICK_ROW:
            return enums.ReasonTrackStart.CLICK_ROW
        elif string == ReasonTrackChange.PLAY_BUTTON:
            return enums.ReasonTrackStart.PLAY_BUTTON
        elif string == ReasonTrackChange.TRACK_ERROR:
            return enums.ReasonTrackStart.TRACK_ERROR
        elif string == ReasonTrackChange.APP_LOAD:
            return enums.ReasonTrackStart.APP_LOAD
        elif string == ReasonTrackChange.POP_UP:
            return enums.ReasonTrackStart.POP_UP
        elif string == ReasonTrackChange.URI_OPEN:
            return enums.ReasonTrackStart.URI_OPEN
        elif string == ReasonTrackChange.CLICK_ROW:
            return enums.ReasonTrackStart.CLICK_ROW
        elif string == ReasonTrackChange.CLICK_SIDE:
            return enums.ReasonTrackStart.CLICK_SIDE
        elif string == ReasonTrackChange.REMOTE:
            return enums.ReasonTrackStart.REMOTE
        else:
            return enums.ReasonTrackStart.UNDEFINED

    @staticmethod
    def get_reason_track_end_from_string(string):
        if string == ReasonTrackChange.FORWARD_BUTTON:
            return enums.ReasonTrackEnd.FORWARD_BUTTON
        elif string == ReasonTrackChange.BACK_BUTTON:
            return enums.ReasonTrackEnd.BACKWARD_BUTTON
        elif string == ReasonTrackChange.TRACK_DONE:
            return enums.ReasonTrackEnd.TRACK_DONE
        elif string == ReasonTrackChange.END_PLAY:
            return enums.ReasonTrackEnd.END_PLAY
        elif string == ReasonTrackChange.LOGOUT:
            return enums.ReasonTrackEnd.LOGOUT
        elif string == ReasonTrackChange.TRACK_ERROR:
            return enums.ReasonTrackEnd.TRACK_ERROR
        elif string == ReasonTrackChange.REMOTE:
            return enums.ReasonTrackEnd.REMOTE
        elif string == ReasonTrackChange.POP_UP:
            return enums.ReasonTrackEnd.POP_UP
        elif string == ReasonTrackChange.APP_LOAD:
            return enums.ReasonTrackEnd.APP_LOAD
        elif string == ReasonTrackChange.URI_OPEN:
            return enums.ReasonTrackEnd.URI_OPEN
        elif string == ReasonTrackChange.CLICK_ROW:
            return enums.ReasonTrackEnd.CLICK_ROW
        elif string == ReasonTrackChange.CLICK_SIDE:
            return enums.ReasonTrackEnd.CLICK_SIDE
        else:
            return enums.ReasonTrackEnd.UNDEFINED
