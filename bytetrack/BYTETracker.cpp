#include "BYTETracker.h"
#include <fstream>

BYTETracker::BYTETracker(int frame_rate, int track_buffer)
{
	track_thresh = 0.5;   // high low det
	high_thresh = 0.6;    // new track
	match_thresh = 1.0;   // cost_limit = match_thresh / 2   匹配loss小于cost_limit的目标

	frame_id = 0;
	max_time_lost = int(frame_rate / 30.0 * track_buffer);
	cout << "Init ByteTrack!" << endl;
}

BYTETracker::~BYTETracker()
{
}

vector<STrack> BYTETracker::update(const vector<Object>& objects)
{
 
	////////////////// Step 1: Get detections //////////////////
	this->frame_id++;
	vector<STrack> activated_stracks;
	vector<STrack> refind_stracks;
	vector<STrack> removed_stracks;
	vector<STrack> lost_stracks;
	vector<STrack> detections;
	vector<STrack> detections_low;

	vector<STrack> detections_cp;
	vector<STrack> tracked_stracks_swap;
	vector<STrack> resa, resb;
	vector<STrack> output_stracks;

	vector<STrack*> unconfirmed;
	vector<STrack*> tracked_stracks;
	vector<STrack*> strack_pool;
	vector<STrack*> r_tracked_stracks;

	if (objects.size() > 0)
	{
		for (int i = 0; i < objects.size(); i++)
		{
			vector<float> tlbr_;
			tlbr_.resize(4);
			tlbr_[0] = objects[i].rect.x;
			tlbr_[1] = objects[i].rect.y;
			tlbr_[2] = objects[i].rect.x + objects[i].rect.width;
			tlbr_[3] = objects[i].rect.y + objects[i].rect.height;

			float score = objects[i].prob;
			float label = objects[i].label;

			STrack strack(STrack::tlbr_to_tlwh(tlbr_), score, label);
			if (score >= track_thresh)
			{
				detections.push_back(strack);
			}
			else
			{
				detections_low.push_back(strack);
			}
			
		}
	}

	// Add newly detected tracklets to tracked_stracks    第一步判断有没有激活
	for (int i = 0; i < this->tracked_stracks.size(); i++)
	{
		if (!this->tracked_stracks[i].is_activated)
			unconfirmed.push_back(&this->tracked_stracks[i]);
		else
			tracked_stracks.push_back(&this->tracked_stracks[i]);
	}

	////////////////// Step 2: First association, with IoU //////////////////
	strack_pool = joint_stracks(tracked_stracks, this->lost_stracks);  // 已激活和失追的目标
	STrack::multi_predict(strack_pool, this->kalman_filter);
	#if TRACK_DEBUG
		std::cout << "Step2" << std::endl;
	#endif
	vector<vector<float> > dists;
	int dist_size = 0, dist_size_size = 0; // strack_pool中轨迹的数量，高置信度detections的数量
	dists = iou_distance(strack_pool, detections, dist_size, dist_size_size);

	vector<vector<int> > matches;
	vector<int> u_track, u_detection;  // 未匹配的预测框 检测框
	linear_assignment(dists, dist_size, dist_size_size, match_thresh, matches, u_track, u_detection);

	for (int i = 0; i < matches.size(); i++)
	{
		STrack *track = strack_pool[matches[i][0]];
		STrack *det = &detections[matches[i][1]];
		if (track->state == TrackState::Tracked)    // 匹配成功 更新状态
		{
			track->update(*det, this->frame_id);
			activated_stracks.push_back(*track);
		}
		else   // 非跟踪状态的且匹配上的框
		{
			track->re_activate(*det, this->frame_id, false);
			refind_stracks.push_back(*track);
		}
	}
	////////////////// Step 2.5: u_track u_detection //////////////////
	vector<STrack> detectionss;
	vector<STrack*> strack_pools;
	for (int i = 0; i < u_detection.size(); i++)
	{
		detectionss.push_back(detections[u_detection[i]]);
	}

	for (int i = 0; i < u_track.size(); i++)
	{
		if (strack_pool[u_track[i]]->state == TrackState::Tracked)   // 判断未匹配预测框是否属于激活状态
		{
			strack_pools.push_back(strack_pool[u_track[i]]);
		}
	}
	#if TRACK_DEBUG
		std::cout << "Step2.5" << std::endl;
	#endif
	dists.clear();
	dists = iou_distance(strack_pools, detectionss, dist_size, dist_size_size); 

	matches.clear();
	u_track.clear();
	u_detection.clear();
	linear_assignment(dists, dist_size, dist_size_size, 1.5, matches, u_track, u_detection);

	for (int i = 0; i < matches.size(); i++)
	{
		STrack *track = strack_pools[matches[i][0]];
		STrack *det = &detectionss[matches[i][1]];
		if (track->state == TrackState::Tracked)    // 匹配成功 更新状态
		{
			track->update(*det, this->frame_id);
			activated_stracks.push_back(*track);
		}
	}

	////////////////// Step 3: Second association, using low score dets //////////////////
	for (int i = 0; i < u_detection.size(); i++)
	{
		detections_cp.push_back(detectionss[u_detection[i]]);  // 存放没有匹配上的high det
	}
	detections.clear();
	detections.assign(detections_low.begin(), detections_low.end());  // 存在low det

	for (int i = 0; i < u_track.size(); i++)
	{
		if (strack_pool[u_track[i]]->state == TrackState::Tracked)   // 判断未匹配预测框是否属于激活状态
		{
			r_tracked_stracks.push_back(strack_pools[u_track[i]]);
		}
	}
	// 3.1
	#if TRACK_DEBUG
		std::cout << "Step3.1" << std::endl;
	#endif
	dists.clear();
	dists = iou_distance(r_tracked_stracks, detections, dist_size, dist_size_size);  // no-match AT && low thresh det

	matches.clear();
	u_track.clear();
	u_detection.clear();
	linear_assignment(dists, dist_size, dist_size_size, 1.4, matches, u_track, u_detection);

	for (int i = 0; i < matches.size(); i++)
	{
		STrack *track = r_tracked_stracks[matches[i][0]];
		STrack *det = &detections[matches[i][1]];
		if (track->state == TrackState::Tracked)    // 二次匹配上更新
		{
			track->update(*det, this->frame_id);
			activated_stracks.push_back(*track);
		}
		else      // 待激活状态
		{
			track->re_activate(*det, this->frame_id, false);
			refind_stracks.push_back(*track);
		}
	}

	// 将还没有匹配上的轨迹的状态标记为Lost，并放入lost_stracks中
	for (int i = 0; i < u_track.size(); i++)
	{        
        STrack *track = r_tracked_stracks[u_track[i]];
		if (track->state != TrackState::Lost)
		{
			track->mark_lost();
			lost_stracks.push_back(*track);
		}
	}
	
	// 3.2
	// Deal with unconfirmed tracks, usually tracks with only one beginning frame   
	// unconfirmed未激活Track NAT && high det
	detections.clear();
	detections.assign(detections_cp.begin(), detections_cp.end());
	#if TRACK_DEBUG
		std::cout << "Step3.2" << std::endl;
	#endif
	dists.clear();
	dists = iou_distance(unconfirmed, detections, dist_size, dist_size_size);

	matches.clear();
	vector<int> u_unconfirmed;
	u_detection.clear();
	linear_assignment(dists, dist_size, dist_size_size, 1.4, matches, u_unconfirmed, u_detection);

	for (int i = 0; i < matches.size(); i++)
	{
		unconfirmed[matches[i][0]]->update(detections[matches[i][1]], this->frame_id);
		activated_stracks.push_back(*unconfirmed[matches[i][0]]);
	}

	for (int i = 0; i < u_unconfirmed.size(); i++)
	{
		STrack *track = unconfirmed[u_unconfirmed[i]];
		track->mark_removed();
		removed_stracks.push_back(*track);
	}

	////////////////// Step 4: Init new stracks //////////////////     
	// 二次还未匹配的high det设为未激活状态   若第一帧则激活状态
	for (int i = 0; i < u_detection.size(); i++)
	{
		STrack *track = &detections[u_detection[i]];
		if (track->score < this->high_thresh)
			continue;
		track->activate(this->kalman_filter, this->frame_id);
		activated_stracks.push_back(*track);
	}

	////////////////// Step 5: Update state //////////////////
	// 不在边界内的删除
	#if DELETE_BORDER
		vector<float> area = {0, 0, 1920, 1080};	
		for (int i = 0; i < this->lost_stracks.size(); i++)    
		{
			if (ious(area, this->lost_stracks[i].tlbr) == 0) 
			{
				// std::cout << this->lost_stracks[i].tlbr[0] << " " << this->lost_stracks[i].tlbr[1] << " " << this->lost_stracks[i].tlbr[2] << " " << this->lost_stracks[i].tlbr[3] << std::endl;
				this->lost_stracks[i].mark_removed();
				removed_stracks.push_back(this->lost_stracks[i]);
			}
		}
	#endif

	// 超过30帧失追目标删除
	for (int i = 0; i < this->lost_stracks.size(); i++)    
	{
		if (this->frame_id - this->lost_stracks[i].end_frame() > this->max_time_lost)
		{
			this->lost_stracks[i].mark_removed();
			removed_stracks.push_back(this->lost_stracks[i]);
		}
	}
	
	// 确保this->tracked_stracks中都为已跟踪状态
	for (int i = 0; i < this->tracked_stracks.size(); i++)
	{
		if (this->tracked_stracks[i].state == TrackState::Tracked)
		{
			tracked_stracks_swap.push_back(this->tracked_stracks[i]);
		}
	}
	this->tracked_stracks.clear();
	this->tracked_stracks.assign(tracked_stracks_swap.begin(), tracked_stracks_swap.end());

	// 合并当前帧匹配成功的AT
	this->tracked_stracks = joint_stracks(this->tracked_stracks, activated_stracks);
	this->tracked_stracks = joint_stracks(this->tracked_stracks, refind_stracks);

	//std::cout << activated_stracks.size() << std::endl;

	this->lost_stracks = sub_stracks(this->lost_stracks, this->tracked_stracks); // 删除重新追踪上的id

	for (int i = 0; i < lost_stracks.size(); i++)
	{
		this->lost_stracks.push_back(lost_stracks[i]);
	}

	this->lost_stracks = sub_stracks(this->lost_stracks, this->removed_stracks);
	this->removed_stracks.clear();
	for (int i = 0; i < removed_stracks.size(); i++)
	{
		this->removed_stracks.push_back(removed_stracks[i]);
	}

	remove_duplicate_stracks(resa, resb, this->tracked_stracks, this->lost_stracks);

	this->tracked_stracks.clear();
	this->tracked_stracks.assign(resa.begin(), resa.end());
	this->lost_stracks.clear();
	this->lost_stracks.assign(resb.begin(), resb.end());
	
	for (int i = 0; i < this->tracked_stracks.size(); i++)
	{
		if (this->tracked_stracks[i].is_activated)
		{
			output_stracks.push_back(this->tracked_stracks[i]);
		}
	}
	return output_stracks;
}