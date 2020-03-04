class DatasetDescription:
    SF_FIRST_HALF = 'session_features_first_half'
    SF_SECOND_HALF = 'session_features_second_half'
    TF_FIRST_HALF = 'track_features_first_half'
    TF_SECOND_HALF = 'track_features_second_half'
    SKIPS = 'skips'


class SessionFeaturesFields:
    SESSION_ID = 'session_id'
    SESSION_LENGTH = 'session_length'
    SESSION_POSITION = 'session_position'
    TRACK_ID_CLEAN = 'track_id_clean'
    SKIP_1 = 'skip_1'
    SKIP_2 = 'skip_2'
    SKIP_3 = 'skip_3'
    NOT_SKIPPED = 'not_skipped'
    CONTEXT_SWITCH = 'context_switch'
    NO_PAUSE_BEFORE_PLAY = 'no_pause_before_play'
    SHORT_PAUSE_BEFORE_PLAY = 'short_pause_before_play'
    LONG_PAUSE_BEFORE_PLAY = 'long_pause_before_play'
    HIST_USER_BEHAVIOR_IS_SHUFFLE = 'hist_user_behavior_is_shuffle'
    HIST_USER_BEHAVIOR_N_SEEKFWD = 'hist_user_behavior_n_seekfwd'
    HIST_USER_BEHAVIOR_N_SEEKBACK = 'hist_user_behavior_n_seekback'
    HIST_USER_BEHAVIOR_REASON_START = 'hist_user_behavior_reason_start'
    HIST_USER_BEHAVIOR_REASON_END = 'hist_user_behavior_reason_end'
    HOUR_OF_DAY = 'hour_of_day'
    DATE = 'date'
    PREMIUM = 'premium'
    CONTEXT_TYPE = 'context_type'


class TrackFeatureFields:
    TRACK_ID = 'track_id'
    DURATION = 'duration'
    RELEASE_YEAR = 'release_year'
    US_POPULARITY_ESTIMATE = 'us_popularity_estimate'
    ACOUSTICNESS = 'acousticness'
    BEAT_STRENGTH = 'beat_strength'
    BOUNCINESS = 'bounciness'
    DANCEABILITY = 'danceability'
    DYN_RANGE_MEAN = 'dyn_range_mean'
    ENERGY = 'energy'
    FLATNESS = 'flatness'
    INSTRUMENTALNESS = 'instrumentalness'
    KEY = 'key'
    LIVENESS = 'liveness'
    LOUDNESS = 'loudness'
    MODE = 'mode'
    ORGANISM = 'organism'
    SPEECHINESS = 'speechiness'
    TEMPO = 'tempo'
    TIME_SIGNATURE = 'time_signature'
    VALENCE = 'valence'
    ACOUSTIC_VECTOR_0 = 'acoustic_vector_0'
    ACOUSTIC_VECTOR_1 = 'acoustic_vector_1'
    ACOUSTIC_VECTOR_2 = 'acoustic_vector_2'
    ACOUSTIC_VECTOR_3 = 'acoustic_vector_3'
    ACOUSTIC_VECTOR_4 = 'acoustic_vector_4'
    ACOUSTIC_VECTOR_5 = 'acoustic_vector_5'
    ACOUSTIC_VECTOR_6 = 'acoustic_vector_6'
    ACOUSTIC_VECTOR_7 = 'acoustic_vector_7'


class SpotifyContextType:
    EDITORIAL_PLAYLIST = 'editorial_playlist'
    CATALOG ="catalog"
    RADIO = "radio"
    CHARTS = "charts"
    USER_COLLECTION ="user_collection"
    PERSONALIZED_PLAYLIST = "personalized_playlist"


class ReasonTrackChange:
    TRACK_DONE = "trackdone"
    FORWARD_BUTTON ="fwdbtn"
    BACK_BUTTON = 'backbtn'
    CLICK_ROW = "clickrow"
    PLAY_BUTTON = "playbtn"
    TRACK_ERROR = "trackerror"
    APP_LOAD = "appload"
    POP_UP = "popup"
    URI_OPEN = "uriopen"
    CLICK_ROW = "clickrow"
    CLICK_SIDE = "clickside"
    REMOTE = "remote"
    END_PLAY = "endplay"
    LOGOUT = "logout"


class TrackMode:
    MINOR = 'minor'
    MAJOR = 'major'
