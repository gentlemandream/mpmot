import itertools
import os
import os.path as osp
import time
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from models import *
from ..models.decode import mot_decode
from ..models.model import create_model, load_model
from ..models.utils import _tranpose_and_gather_feat
from ..tracking_utils.kalman_filter import KalmanFilter
from ..tracking_utils.log import logger
from ..tracking_utils.utils import *
from ..utils.image import get_affine_transform
from ..utils.post_process import ctdet_post_process

from ..tracker import matching

from .basetrack import BaseTrack, TrackState


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, temp_feat, buffer_size=30):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        # self.score_list = []
        self.tracklet_len = 0

        self.smooth_feat = None
        self.update_features(temp_feat)
        self.features = deque([], maxlen=buffer_size)
        self.alpha = 0.9

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        #self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

        # self.score_list.append(self.score)

    def re_activate(self, new_track, frame_id, new_id=False, flag=.0):
        # print(new_track.score)  NSA只在这用 比另外好, new_track.score
        if new_track.score >= 0.45:
            con = 0
        else:
            con = new_track.score
        # if(flag == 1):
        #     # if new_track.score >= 0.6:
        #     #     con = 0.0
        #     # else:
        #     #     con = new_track.score
        #     self.mean, self.covariance = self.kalman_filter.update(
        #         self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh), new_track.score
        #     )  # +置信度
        # else:
        #     self.mean, self.covariance = self.kalman_filter.update(
        #         self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        #     )  # +置信度, new_track.score
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh), con
        )  # +置信度
        self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

        # self.score = new_track.score
        # self.score_list.append(self.score)
    #0.6 69.1 72.5
    #update emb+iou+uncon  activate emb+iou
    def update(self, new_track, frame_id, update_feature=True, flag=.0): #detections[idet], self.frame_id #dets[:, 4]表示置信度
        """
        Update a matched track
        :type new_track: STrack  [tlwh,score,feature,30]
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        # if new_track.score >=0.56:#44 43 42
        #     con = 0.0
        # else:
        #     con = new_track.score
        if(flag == 1):
            if new_track.score >= 0.56:  # 44 43 42
                con = 0
            else:
                con = new_track.score
            self.mean, self.covariance = self.kalman_filter.update(
                self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh), con)
        else:
            self.mean, self.covariance = self.kalman_filter.update(
                self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        # self.mean, self.covariance = self.kalman_filter.update(
        #     self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh), new_track.score)
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

        # self.score_list.append(self.score)

        if update_feature:
            self.update_features(new_track.curr_feat)

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class JDETracker(object):
    def __init__(self, opt, frame_rate=30):
        self.opt = opt
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')
        print('Creating model...')
        self.model = create_model(opt.arch, opt.heads, opt.head_conv)
        self.model = load_model(self.model, opt.load_model)
        self.model = self.model.to(opt.device)
        self.model.eval()

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.det_thresh = opt.conf_thres + 0.16  #0.1~0.2 11,12,13,14,15   0.05 0.025 0.075  75 /85/95
        self.buffer_size = int(frame_rate / 30.0 * opt.track_buffer)
        self.max_time_lost = self.buffer_size
        self.max_per_image = opt.K
        self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)

        self.kalman_filter = KalmanFilter()

    def post_process(self, dets, meta):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], self.opt.num_classes)
        for j in range(1, self.opt.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
        return dets[0]

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.opt.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)

        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.opt.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.opt.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    def update(self, im_blob, img0):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        width = img0.shape[1]
        height = img0.shape[0]   #原图wh
        inp_height = im_blob.shape[2] #加工后 wh
        inp_width = im_blob.shape[3]
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(float(inp_width) / float(inp_height) * height, width) * 1.0  #w宽
        meta = {'c': c, 's': s,
                'out_height': inp_height // self.opt.down_ratio,   #输出wh为加工后wh/缩小比列
                'out_width': inp_width // self.opt.down_ratio}

        ''' Step 1: Network forward, get detections & embeddings'''
        with torch.no_grad(): #最好不改
            output = self.model(im_blob)[-1]   #对模型输入图片
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            id_feature = output['id']
            id_feature = F.normalize(id_feature, dim=1)

            reg = output['reg'] if self.opt.reg_offset else None
            dets, inds = mot_decode(hm, wh, reg=reg, ltrb=self.opt.ltrb, K=self.opt.K)
            id_feature = _tranpose_and_gather_feat(id_feature, inds)
            id_feature = id_feature.squeeze(0)
            id_feature = id_feature.cpu().numpy()

        dets = self.post_process(dets, meta)
        dets = self.merge_outputs([dets])[1]

        remain_inds = dets[:, 4] > self.opt.conf_thres  #dets[:, 4]表示置信度0.4

        # #保留一部分低分 检测框和对应的特征
        # inds_low = dets[:, 4] > 0.2  #0.2   25,30,35
        # inds_high = dets[:, 4] < self.opt.conf_thres#0.4
        # inds_second = np.logical_and(inds_low, inds_high)  # 0.2-0.4  存在为1
        # dets_second = dets[inds_second]  # 0.2-0.4这一部分 检测框 给dets_second
        # id_feature_second = id_feature[inds_second]

        dets = dets[remain_inds]
        id_feature = id_feature[remain_inds]

        # vis  检测结果可视化
        # cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        # for i in range(0, dets.shape[0]):
        #     bbox = dets[i][0:4]
        #     # img1=np.ascontiguousarray(np.copy(img0))
        #     cv2.rectangle(img0, (bbox[0], bbox[1]),
        #                   (bbox[2], bbox[3]),
        #                   (0, 255, 0), 2)
        # cv2.imshow('dets', img0)
        # cv2.waitKey(0)
        # # id0 = id0-1


        if len(dets) > 0:
            '''Detections'''#tlwh, score, temp_feat, buffer_size=30
            detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30) for
                          (tlbrs, f) in zip(dets[:, :5], id_feature)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with embedding'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        #for strack in strack_pool:
            #strack.predict()
        STrack.multi_predict(strack_pool)

        # ious_dists = matching.iou_distance(strack_pool, detections)  # 无交集iou 就为1  计算为多个框
        # ious_dists_mask = (ious_dists > 0.4)
        # cos_dists = matching.embedding_distance(strack_pool, detections)
        # emb_dists = cos_dists / 2.0  # *0.5
        # emb_dists[emb_dists > 0.25] = 1.0
        # emb_dists[ious_dists_mask] = 1.0
        # dists = np.minimum(ious_dists, emb_dists)
        # dists1 = matching.fuse_motion(self.kalman_filter, cos_dists, strack_pool, detections)
        # dists[ious_dists_mask] = dists1[ious_dists_mask]

        # match_thresh=0.4
        ious_dists = matching.iou_distance(strack_pool, detections)
        ious_dists_mask = (ious_dists > 0.4)  #0.6
        cos_dists = matching.embedding_distance(strack_pool, detections)
        fu_dists = matching.fuse_motion(self.kalman_filter, cos_dists, strack_pool, detections)
        # 不除2
        fu_dists[cos_dists > 0.5] = 1.0
        dists = np.minimum(ious_dists, fu_dists)

        # ious_dists = matching.iou_distance(strack_pool, detections)
        # ious_dists_mask = (ious_dists > 0.5)  # iou>0.5
        #
        # emb_dists = matching.embedding_distance(strack_pool, detections) / 2.0  # *0.5
        # raw_emb_dists = emb_dists.copy()
        # emb_dists[emb_dists > 0.25] = 1.0  # cos>0.25  iou>0.5   cos =1
        # emb_dists[ious_dists_mask] = 1.0
        # dists = np.minimum(ious_dists, emb_dists)

        # dists = matching.embedding_distance(strack_pool, detections)
        # #dists = matching.iou_distance(strack_pool, detections)
        # dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.4)#0.7

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id, flag=1)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)  #, flag=1
                refind_stracks.append(track)

        ''' Step 3: Second association, with IOU'''
        detections = [detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # association the untrack to the low score detections  72.5/75.0
        # if len(dets_second) > 0:
        #         '''Detections'''
        #         detections_second = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30) for
        #                              (tlbrs, f) in zip(dets_second[:, :5], id_feature_second)]
        # else:
        #     detections_second = []
        # # r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        # second_tracked_stracks = [r_tracked_stracks[i] for i in u_track if r_tracked_stracks[i].state == TrackState.Tracked]
        #
        # ious_dists = matching.iou_distance(second_tracked_stracks, detections_second)
        # # ious_dists_mask = (ious_dists > 0.4)  #0.6
        # cos_dists = matching.embedding_distance(second_tracked_stracks, detections_second)
        # fu_dists = matching.fuse_motion(self.kalman_filter, cos_dists, second_tracked_stracks, detections_second)
        # # 不除2
        # fu_dists[cos_dists > 0.5] = 1.0
        # dists = np.minimum(ious_dists, fu_dists)
        #
        # # dists = matching.iou_distance(second_tracked_stracks, detections_second)
        # matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5) #0.4  30,35,45 31,32,33,34
        # for itracked, idet in matches:
        #         track = second_tracked_stracks[itracked]
        #         det = detections_second[idet]
        #         if track.state == TrackState.Tracked:
        #             track.update(det, self.frame_id)
        #             activated_starcks.append(track)
        #         else:
        #             track.re_activate(det, self.frame_id, new_id=False)
        #             refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            # track = second_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)

        # ious_dists =matching.iou_distance(unconfirmed, detections)
        # cos_dists = matching.embedding_distance(unconfirmed, detections)
        # fu_dists = matching.fuse_motion(self.kalman_filter, cos_dists, unconfirmed, detections)
        # fu_dists[cos_dists > 0.5] = 1.0
        # dists = np.minimum(ious_dists, fu_dists)

        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])

        # if len(dets_second) > 0:
        #     '''Detections'''
        #     detections_second = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30) for
        #                          (tlbrs, f) in zip(dets_second[:, :5], id_feature_second)]
        # else:
        #     detections_second = []
        # # r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        # second_tracked_stracks = [unconfirmed[i] for i in u_unconfirmed]
        # dists = matching.iou_distance(second_tracked_stracks, detections_second)
        # matches, u_unconfirmed, u = matching.linear_assignment(dists, thresh=0.6)  #4，5，6
        # for itracked, idet in matches:
        #     second_tracked_stracks[itracked].update(detections_second[idet], self.frame_id)
        #     activated_starcks.append(second_tracked_stracks[itracked])

        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        logger.debug('===========Frame {}=========='.format(self.frame_id))
        logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
        logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))

        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
