import motmetrics as mot
import numpy as np
import warnings
import ipdb

# each track is a dictionary with car-ids as dictionary ID
# same for gt tracks
# validate is to be called for every frame


class tracker_evaluator:

    def __init__(self, max_dist):
        self._acc = mot.MOTAccumulator(auto_id=True)
        self._max_dist = max_dist  # norm2sqr_matrix requires square of max_dist


    def validate(self, tracks, gt_tracks):
        # {id1 : x, y, vx, vy, ax ,ay, id2 : x, y, ..}
        # [[x, y, vx, vy, ax, ay, id1], [x, y, vx, vy, ax, ay, id2]]
        detections = np.asarray([tracks[id][:2] for id in tracks])
        grnd_truth = np.asarray([gt_tracks[id][:2] for id in gt_tracks])
        cost_matrix = np.sqrt(mot.distances.norm2squared_matrix(grnd_truth, detections, max_d2=(self._max_dist*self._max_dist)))
        # print(cost_matrix)
        # print(detections)
        # print(grnd_truth)
        if(gt_tracks.keys()):
            self._acc.update(np.asarray(list(gt_tracks.keys())), np.asarray(list(tracks.keys())), cost_matrix)


    def full_metrics(self):

        # print accuracy results
        warnings.filterwarnings("ignore", category=FutureWarning)
        # print('\n\033[91m' + '*' * 30 + ' MOT Metrics Summary ' + '*' * 30 + '\n')
        print( '-' * 30 + ' MOT Metrics Summary ' + '-' * 30)
        metrics = mot.metrics.create()
        summary = metrics.compute(self._acc, metrics=mot.metrics.motchallenge_metrics, name='Overall')
        print(mot.io.render_summary(summary, formatters=metrics.formatters,
                                    namemap=mot.io.motchallenge_metric_names))
        # ipdb.set_trace()


    def mot_metrics(self):

        # print accuracy results
        warnings.filterwarnings("ignore", category=FutureWarning)
        metrics = mot.metrics.create()
        summary = metrics.compute(self._acc, metrics=['num_frames', 'mota', 'motp'], name='acc')
        mota = summary['mota'][0]*100 # to percent
        #TODO : make generic
        # motp = (1 - summary['motp'][0]/100)*100
        motp = (1 - summary['motp'][0]/self._max_dist)*100
        # str_summ = mot.io.render_summary(summary,
        #                       formatters=metrics.formatters,
        #                       namemap={'mota': 'MOTA', 'motp': 'MOTP'})
        return mota, motp
