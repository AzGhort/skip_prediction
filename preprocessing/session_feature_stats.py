from dataset_description import *
from enums import *
import numpy as np


Maximums = {
    SessionFeaturesFields.CONTEXT_TYPE : SpotifyContextType.PERSONALIZED_PLAYLIST,
    SessionFeaturesFields.HIST_USER_BEHAVIOR_REASON_START: ReasonTrackStart.END_PLAY,
    SessionFeaturesFields.HIST_USER_BEHAVIOR_REASON_END: ReasonTrackEnd.CLICK_ROW,
    SessionFeaturesFields.HOUR_OF_DAY: 23.0,
    SessionFeaturesFields.HIST_USER_BEHAVIOR_N_SEEKFWD: 51458.0,
    SessionFeaturesFields.HIST_USER_BEHAVIOR_N_SEEKBACK: 88162.0
}

LogMaximums = {
    SessionFeaturesFields.HIST_USER_BEHAVIOR_N_SEEKFWD: np.log(51458.0),
    SessionFeaturesFields.HIST_USER_BEHAVIOR_N_SEEKBACK: np.log(88162.0)
}


