class SpotifyContextType:
    UNDEFINED = 0.
    EDITORIAL_PLAYLIST = 1.
    USER_COLLECTION = 2.
    CHARTS = 3.
    RADIO = 4.
    CATALOG = 5.
    PERSONALIZED_PLAYLIST = 6.


class ReasonTrackStart:
    UNDEFINED = 0.
    FORWARD_BUTTON = 1.
    BACKWARD_BUTTON = 2.
    TRACK_DONE = 3.
    APP_LOAD = 4.
    CLICK_ROW = 5.
    PLAY_BUTTON = 6.
    TRACK_ERROR = 7.
    POP_UP = 8.
    CLICK_SIDE = 9.
    REMOTE = 10.
    URI_OPEN = 11.
    END_PLAY = 12.


class ReasonTrackEnd:
    UNDEFINED = 0.
    FORWARD_BUTTON = 1.
    BACKWARD_BUTTON = 2.
    TRACK_DONE = 3.
    LOGOUT = 4.
    END_PLAY = 5.
    REMOTE = 6.
    TRACK_ERROR = 7.
    APP_LOAD = 8.
    CLICK_SIDE = 9.
    POP_UP = 10.
    URI_OPEN = 11.
    CLICK_ROW = 12.


class PredictionMode:
    UNDEFINED = 0.
    TRACK_FEATURES = 1.
    LOG_FEATURES = 2.
    ALL_FEATURES = 3.


class TrackMode:
    MINOR = 0.
    MAJOR = 1.
