import csv
import os

skips = {'skip_1': 0, 'skip_2': 0, 'skip_3': 0, 'not_skipped': 0}
track_ids = {}
session_lengths = {i: 0 for i in range(21)}
premiums = 0
total_count = 0

context = {}
reason_track_start = {}
reason_track_end = {}

for filename in os.listdir("."):
    print('processing file ' + filename)
    if filename.endswith('.csv'):
        with open(filename) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['skip_1'] == 'true':
                    skips['skip_1'] += 1
                elif row['skip_2'] == 'true':
                    skips['skip_2'] += 1
                elif row['skip_3'] == 'true':
                    skips['skip_3'] += 1
                else:
                    skips['not_skipped'] += 1

                if row['premium'] == 'true':
                    premiums += 1
                if row['track_id_clean'] in track_ids:
                    track_ids[row['track_id_clean']] += 1
                else:
                    track_ids[row['track_id_clean']] = 1

                if row['hist_user_behavior_reason_start'] in reason_track_start:
                    reason_track_start[row['hist_user_behavior_reason_start']] += 1
                else:
                    reason_track_start[row['hist_user_behavior_reason_start']] = 1

                if row['hist_user_behavior_reason_end'] in reason_track_end:
                    reason_track_end[row['hist_user_behavior_reason_end']] += 1
                else:
                    reason_track_end[row['hist_user_behavior_reason_end']] = 1

                if row['context_type'] in context:
                    context[row['context_type']] += 1
                else:
                    context[row['context_type']] = 1

                total_count += 1

skps = open('skips.txt', 'w+')
for key in skips:
    skps.write(key + ' ' + str(skips[key]))
    skps.write(os.linesep)
skps.close()

r_start = open('reason_start.txt', 'w+')
for key in reason_track_start:
    r_start.write(key + ' ' + str(reason_track_start[key]))
    r_start.write(os.linesep)
r_start.close()

r_end = open('reason_end.txt', 'w+')
for key in reason_track_end:
    r_end.write(key + ' ' + str(reason_track_end[key]))
    r_end.write(os.linesep)
r_end.close()

ctx = open('context.txt', 'w+')
for key in context:
    ctx.write(key + ' ' + str(context[key]))
    ctx.write(os.linesep)
ctx.close()

lngths = open('session_lengths.txt', 'w+')
for key in session_lengths:
    lngths.write(str(key) + ' ' + str(session_lengths[key]))
    lngths.write(os.linesep)
lngths.close()

premium = open('premium.txt', 'w+')
premium.write('premium ' + str(premiums))
premium.write(os.linesep)
premium.write('free ' + str((total_count - premiums)))
premium.write(os.linesep)
premium.close()

print('done')



