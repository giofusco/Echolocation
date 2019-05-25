import cv2
from utils.progress_bar import print_progress_bar
from cv_tools import marker_detector as md
from utils import analysis


scale = 2025 # pixels x meter @ 1635 x 1053

video = 'GOPR3956.MP4'

video_src = '../../Subj01_DK/Video/' + video
center_file = './results/' + video + '_center.txt'
log_file = './results/' + video + '_log.txt'

log = open(log_file, 'w')
log_header = 'ts \t trial # \t led \t alpha \t gamma \t theta \t target_pos \t center_pos \t pointer_pt \n'
log.write(log_header)

center_ratio = analysis.find_center(video_src, center_file)

cap = cv2.VideoCapture(video_src)
num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS)

marker_det = md.MarkerDetector()

drop_cnt = 0
trial_started = False
trial_start_threshold = 30
ready_to_start = False
trial_cnt = 0
frame_cnt = 0
tstamp = 0

prev_brightness = -1
led_delta = -1
led_triggered = False

final_gt_angle = 0
final_head_angle = 0



while True:

    ret, frame = cap.read()
    frame_cnt += 1
    if frame is not None:

        tstamp = 1 / fps * frame_cnt

        vis, alpha, gamma, theta, pt_target, pt_head, pt_nose = analysis.process_frame(frame, center_ratio, marker_det,
                                                                                       scale)
        cv2.rectangle(vis, (500, 280), (560, 320), (0, 255, 0), 2)

        if alpha is not None:
            drop_cnt = 0
            if ready_to_start:
                ready_to_start = False
                trial_started = True

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (int(frame.shape[1] / 3), int(frame.shape[0] / 3)))
            led_quad = gray[280:320, 500:560]

            if prev_brightness < 0:
                prev_brightness = cv2.mean(led_quad)[0]
                led_delta = 0
            else:
                curr_brightness = cv2.mean(led_quad)[0]
                led_delta = curr_brightness - prev_brightness
                if led_delta > 30:
                    cv2.rectangle(frame, (500, 280), (560, 320), (0, 0, 255), 2)
                    trial_started = False
                    led_triggered = True
                else:
                    prev_brightness = curr_brightness
                    led_triggered = False
        else:
            drop_cnt += 1

            if (drop_cnt > trial_start_threshold) and (ready_to_start is False):
                trial_cnt += 1
                ready_to_start = True

        log_line = str(tstamp) + '\t' + str(trial_cnt) + '\t' + str(led_triggered) + '\t' + str(alpha) + '\t' + \
                   str(gamma) + '\t' + str(theta) + '\t' + str(pt_target) + '\t' + str(pt_head) + '\t' + str(pt_nose) \
                   + '\n'

        if trial_started:

            log.write(log_line)
            header = "trial #" + str(trial_cnt)
            cv2.putText(vis, header, (10, 20), cv2.FONT_HERSHEY_DUPLEX,
                         .45, (125, 125, 0))

        elif led_triggered: # save answer and pause logging
            log.write(log_line)
            header = "trial #" + str(trial_cnt) + " gt_angle: " + str(alpha)
            cv2.putText(vis, header, (10, 20), cv2.FONT_HERSHEY_DUPLEX,
                        .45, (125, 125, 0))
            ans = "ans: " + str(gamma)
            cv2.putText(vis, ans, (10, 50), cv2.FONT_HERSHEY_DUPLEX,
                        .45, (125, 125, 0))
            led_triggered = False
            trial_started = False
            ready_to_start = False

        elif trial_started is False:
            cv2.putText(vis, "waiting for trial start...", (10, 20), cv2.FONT_HERSHEY_DUPLEX,
                        .45, (125, 125, 0))

        print_progress_bar(frame_cnt + 1, num_frames, prefix='Progress:',
                           suffix='Complete', length=50)

        cv2.imshow("Frame", vis)
        cv2.waitKey(1)
    if frame_cnt == num_frames:
        print("\n All done, bye.")
        break
