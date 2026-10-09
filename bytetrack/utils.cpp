#include "BYTETracker.h"
#include "lapjv.h"

vector<STrack*> BYTETracker::joint_stracks(vector<STrack*> &tlista, vector<STrack> &tlistb)
{
	map<int, int> exists;
	vector<STrack*> res;
	for (int i = 0; i < tlista.size(); i++)
	{
		exists.insert(pair<int, int>(tlista[i]->track_id, 1));
		res.push_back(tlista[i]);
	}
	for (int i = 0; i < tlistb.size(); i++)
	{
		int tid = tlistb[i].track_id;
		if (!exists[tid] || exists.count(tid) == 0)
		{
			exists[tid] = 1;
			res.push_back(&tlistb[i]);
		}
	}
	return res;
}

vector<STrack> BYTETracker::joint_stracks(vector<STrack> &tlista, vector<STrack> &tlistb)
{
	map<int, int> exists;
	vector<STrack> res;
	for (int i = 0; i < tlista.size(); i++)
	{
		exists.insert(pair<int, int>(tlista[i].track_id, 1));
		res.push_back(tlista[i]);
	}
	for (int i = 0; i < tlistb.size(); i++)
	{
		int tid = tlistb[i].track_id;
		if (!exists[tid] || exists.count(tid) == 0)
		{
			exists[tid] = 1;
			res.push_back(tlistb[i]);
		}
	}
	return res;
}

vector<STrack> BYTETracker::sub_stracks(vector<STrack> &tlista, vector<STrack> &tlistb)
{
	map<int, STrack> stracks;
	for (int i = 0; i < tlista.size(); i++)
	{
		stracks.insert(pair<int, STrack>(tlista[i].track_id, tlista[i]));
	}
	for (int i = 0; i < tlistb.size(); i++)
	{
		int tid = tlistb[i].track_id;
		if (stracks.count(tid) != 0)
		{
			stracks.erase(tid);
		}
	}

	vector<STrack> res;
	std::map<int, STrack>::iterator  it;
	for (it = stracks.begin(); it != stracks.end(); ++it)
	{
		res.push_back(it->second);
	}

	return res;
}

void BYTETracker::remove_duplicate_stracks(vector<STrack> &resa, vector<STrack> &resb, vector<STrack> &stracksa, vector<STrack> &stracksb)
{
	vector<vector<float> > pdist = iou_distance(stracksa, stracksb);
	vector<pair<int, int> > pairs;
	for (int i = 0; i < pdist.size(); i++)
	{
		for (int j = 0; j < pdist[i].size(); j++)
		{
			if (pdist[i][j] < 0.15)
			{
				pairs.push_back(pair<int, int>(i, j));
			}
		}
	}
	// 将时间短的轨迹标记为重复轨迹
	unordered_set<int> dupa, dupb;
	for (int i = 0; i < pairs.size(); i++) 
	{
		int timep = stracksa[pairs[i].first].frame_id - stracksa[pairs[i].first].start_frame;
		int timeq = stracksb[pairs[i].second].frame_id - stracksb[pairs[i].second].start_frame;
		if (timep > timeq)
			dupb.insert(pairs[i].second);
		else
			dupa.insert(pairs[i].first);
	}

	for (int i = 0; i < stracksa.size(); i++) 
	{
		if (dupa.find(i) == dupa.end()) 
		{
			resa.push_back(stracksa[i]);
		}
	}

	for (int i = 0; i < stracksb.size(); i++) 
	{
		if (dupb.find(i) == dupb.end()) 
		{
			resb.push_back(stracksb[i]);
		}
	}
}

void BYTETracker::linear_assignment(vector<vector<float> > &cost_matrix, int cost_matrix_size, int cost_matrix_size_size, float thresh,
	vector<vector<int> > &matches, vector<int> &unmatched_a, vector<int> &unmatched_b)
{
	if (cost_matrix.size() == 0)
	{
		for (int i = 0; i < cost_matrix_size; i++)
		{
			unmatched_a.push_back(i);
		}
		for (int i = 0; i < cost_matrix_size_size; i++)
		{
			unmatched_b.push_back(i);
		}
		return;
	}

	vector<int> rowsol; vector<int> colsol;
	float c = lapjv(cost_matrix, rowsol, colsol, true, thresh);
	for (int i = 0; i < rowsol.size(); i++)
	{
		if (rowsol[i] >= 0)
		{
			vector<int> match;    // (match index)
			match.push_back(i);
			match.push_back(rowsol[i]);
			matches.push_back(match);
		}
		else
		{
			unmatched_a.push_back(i);
		}
	}

	for (int i = 0; i < colsol.size(); i++)
	{
		if (colsol[i] < 0)
		{
			unmatched_b.push_back(i);
		}
	}
}

float BYTETracker::ious(vector<float> atlbrs, vector<float> btlbrs)
{
    float interBox[] = {
        (std::max)(atlbrs[0], btlbrs[0]), //left
        (std::min)(atlbrs[2], btlbrs[2]), //right
        (std::max)(atlbrs[1], btlbrs[1]), //top
        (std::min)(atlbrs[3], btlbrs[3]), //bottom
    };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
	float aw = atlbrs[2] - atlbrs[0];
	float ah = atlbrs[3] - atlbrs[1];
	float bw = btlbrs[2] - btlbrs[0];
	float bh = btlbrs[3] - btlbrs[1];

    return interBoxS / (aw * ah + bw * bh - interBoxS);
}


vector<vector<float> > BYTETracker::ious(vector<vector<float> > &atlbrs, vector<vector<float> > &btlbrs)
{
	vector<vector<float>> ious;
	if (atlbrs.size()*btlbrs.size() == 0)
		return ious;

	ious.resize(atlbrs.size());
	for (int i = 0; i < ious.size(); i++)
	{
		ious[i].resize(btlbrs.size());
	}

	//bbox_ious
	for (int k = 0; k < btlbrs.size(); k++)
	{
		float box_area1 = (btlbrs[k][2] - btlbrs[k][0] + 1)*(btlbrs[k][3] - btlbrs[k][1] + 1);
        float cx1 = (btlbrs[k][0] + btlbrs[k][2]) / 2.0f;
        float cy1 = (btlbrs[k][1] + btlbrs[k][3]) / 2.0f;
		float w1 = btlbrs[k][2] - btlbrs[k][0];
		float h1 = btlbrs[k][3] - btlbrs[k][1];
		float ratio1 = w1 / h1;
		for (int n = 0; n < atlbrs.size(); n++)
		{
			
			#if JUDGE_CLASS
				if ((btlbrs[k][4] == 0 || btlbrs[k][4] == 1 || btlbrs[k][4] == 2) && (atlbrs[n][4] == 3 || atlbrs[n][4] == 4)) {
					ious[n][k] = -1.0;
					continue;
				}

				if ((btlbrs[k][4] == 3 || btlbrs[k][4] == 4) && (atlbrs[n][4] == 0 || atlbrs[n][4] == 1 || atlbrs[n][4] == 2)) {
					ious[n][k] = -1.0;
					continue;
				}
			#endif
				
			float iw = min(atlbrs[n][2], btlbrs[k][2]) - max(atlbrs[n][0], btlbrs[k][0]);
		
			float box_area2 = (atlbrs[n][2] - atlbrs[n][0] + 1)*(atlbrs[n][3] - atlbrs[n][1] + 1);
            float cx2 = (atlbrs[n][0] + atlbrs[n][2]) / 2.0f;
            float cy2 = (atlbrs[n][1] + atlbrs[n][3]) / 2.0f;
			float w2 = atlbrs[n][2] - atlbrs[n][0];
			float h2 = atlbrs[n][3] - atlbrs[n][1];		
			float ratio2 = w2 / h2;

			// dious	
            float distance_squared = (cx1 - cx2) * (cx1 - cx2) + (cy1 - cy2) * (cy1 - cy2);
            float cw = std::max(btlbrs[k][2], atlbrs[n][2]) - std::min(btlbrs[k][0], atlbrs[n][0]);
            float ch = std::max(btlbrs[k][3], atlbrs[n][3]) - std::min(btlbrs[k][1], atlbrs[n][1]);
            float c_squared = cw * cw + ch * ch;

			float ratio = std::min(ratio1 / ratio2, ratio2 / ratio1);
			float area = std::min(box_area1 / box_area2, box_area2 / box_area1);

			if (iw > 0)
			{
				float ih = min(atlbrs[n][3], btlbrs[k][3]) - max(atlbrs[n][1], btlbrs[k][1]);
				if(ih > 0)
				{
					float ua = box_area1 + box_area2 - iw * ih;
					ious[n][k] = iw * ih / ua - distance_squared / c_squared;
				}
				else if (distance_squared / c_squared < 0.8 && area > 0.5 && ratio > 0.4) 
				{
					ious[n][k] = 0.0 - distance_squared / c_squared;
				}
				else
				{
					ious[n][k] = -1.0;
				}
			}
			else if (distance_squared / c_squared < 0.8 && area > 0.5 && ratio > 0.4) 
			{
				ious[n][k] = 0.0 - distance_squared / c_squared;
			}
			else
			{
				ious[n][k] = -1.0;
			}
		}
	}

	return ious;
}

vector<vector<float> > BYTETracker::iou_distance(vector<STrack*> &atracks, vector<STrack> &btracks, int &dist_size, int &dist_size_size)
{
	vector<vector<float> > cost_matrix;
	if (atracks.size() * btracks.size() == 0)
	{
		dist_size = atracks.size();
		dist_size_size = btracks.size();
		return cost_matrix;
	}
	vector<vector<float> > atlbrs, btlbrs;
	vector<float> atlbr, btlbr;
	for (int i = 0; i < atracks.size(); i++)
	{
		atlbr = atracks[i]->tlbr;
		atlbr.push_back(atracks[i]->label);
		atlbrs.push_back(atlbr);
	}
	for (int i = 0; i < btracks.size(); i++)
	{
		btlbr = btracks[i].tlbr;
		btlbr.push_back(btracks[i].label);
		btlbrs.push_back(btlbr);
	}

	#if TRACK_DEBUG
		// 配对
		std::cout << "预测tl" << std::endl;
		for (int i = 0; i < atracks.size(); i++)
		{
			std::cout << "(" << i << ":" << atracks[i]->tlwh[0] << "," << atracks[i]->tlwh[1] << "," << atracks[i]->tlwh[2] << "," << atracks[i]->tlwh[3] <<")";
			
		}
		std::cout << std::endl;

		std::cout << "检测tl" << std::endl;
		for (int i = 0; i < btracks.size(); i++)
		{
			std::cout << "(" << i << ":" << btracks[i].tlwh[0] << "," << btracks[i].tlwh[1] << "," << btracks[i].tlwh[2] << "," << btracks[i].tlwh[3] << ")";
			
		}
		std::cout << std::endl;
	#endif

	dist_size = atracks.size();
	dist_size_size = btracks.size();

	vector<vector<float> > _ious = ious(atlbrs, btlbrs);
	
	for (int i = 0; i < _ious.size();i++)
	{
		vector<float> _iou;
		for (int j = 0; j < _ious[i].size(); j++)
		{
			_iou.push_back(1 - _ious[i][j]);
		}
		cost_matrix.push_back(_iou);
	}

	return cost_matrix;
}

vector<vector<float> > BYTETracker::iou_distance(vector<STrack> &atracks, vector<STrack> &btracks)
{
	vector<vector<float> > atlbrs, btlbrs;
	for (int i = 0; i < atracks.size(); i++)
	{
		atlbrs.push_back(atracks[i].tlbr);
	}
	for (int i = 0; i < btracks.size(); i++)
	{
		btlbrs.push_back(btracks[i].tlbr);
	}

	vector<vector<float> > _ious = ious(atlbrs, btlbrs);
	vector<vector<float> > cost_matrix;
	for (int i = 0; i < _ious.size(); i++)
	{
		vector<float> _iou;
		for (int j = 0; j < _ious[i].size(); j++)
		{
			_iou.push_back(1 - _ious[i][j]);
		}
		cost_matrix.push_back(_iou);
	}

	return cost_matrix;
}


double BYTETracker::lapjv(const vector<vector<float> > &cost, vector<int> &rowsol, vector<int> &colsol,
	bool extend_cost, float cost_limit, bool return_cost)  // cost_limit 成本限制，超过此值的分配将被忽略
{
	vector<vector<float> > cost_c;
	cost_c.assign(cost.begin(), cost.end());

	vector<vector<float> > cost_c_extended;

	int n_rows = cost.size();
	int n_cols = cost[0].size();
	rowsol.resize(n_rows);
	colsol.resize(n_cols);

	int n = 0;
	if (n_rows == n_cols)
	{
		n = n_rows;
	}
	else
	{
		if (!extend_cost)
		{
			cout << "set extend_cost=True" << endl;
			system("pause");
			exit(0);
		}
	}
	// 扩展成本矩阵 为了处理任务数和代理数不相等的情况  extend_cost是否拓展
	if (extend_cost || cost_limit < LONG_MAX)
	{
		n = n_rows + n_cols;
		cost_c_extended.resize(n);
		for (int i = 0; i < cost_c_extended.size(); i++)
			cost_c_extended[i].resize(n);

		if (cost_limit < LONG_MAX)
		{
			for (int i = 0; i < cost_c_extended.size(); i++)
			{
				for (int j = 0; j < cost_c_extended[i].size(); j++)
				{
					cost_c_extended[i][j] = cost_limit / 2.0;   // thresh / 2
				}
			}
		}
		else
		{
			float cost_max = -1;
			for (int i = 0; i < cost_c.size(); i++)
			{
				for (int j = 0; j < cost_c[i].size(); j++)
				{
					if (cost_c[i][j] > cost_max)
						cost_max = cost_c[i][j];
				}
			}
			for (int i = 0; i < cost_c_extended.size(); i++)
			{
				for (int j = 0; j < cost_c_extended[i].size(); j++)
				{
					cost_c_extended[i][j] = cost_max + 1;   // 将虚拟成本设置为原始矩阵中的最大值加 1
				}
			}
		}

		// 0~n = cost_c[i][j]  n~maxsize()=0 虚拟任务和虚拟代理之间的成本设置为 0，表示这些分配不会影响总成本
		for (int i = n_rows; i < cost_c_extended.size(); i++)
		{
			for (int j = n_cols; j < cost_c_extended[i].size(); j++)
			{
				cost_c_extended[i][j] = 0;
			}
		}
		for (int i = 0; i < n_rows; i++)
		{
			for (int j = 0; j < n_cols; j++)
			{
				cost_c_extended[i][j] = cost_c[i][j];
			}
		}

		#if TRACK_DEBUG
			// 代价矩阵		
			std::cout << "代价矩阵" << std::endl;
			for (int i = 0; i < cost_c_extended.size(); i++)
			{
				for (int j = 0; j < cost_c_extended.size(); j++)
				{
					std::cout << cost_c_extended[i][j] << " ";
				}
				std::cout << std::endl;	
			}
		#endif

		cost_c.clear();
		cost_c.assign(cost_c_extended.begin(), cost_c_extended.end());
	}

	// 将C++的二维向量（vector<vector<float>>）转换为C风格的二维指针（double**）以便与底层C函数（如 lapjv_internal）兼容
	// 确保内存布局连续，提高访问效率。兼容底层 C 函数的输入要求。
	double **cost_ptr;
	cost_ptr = new double *[sizeof(double *) * n];
	for (int i = 0; i < n; i++)
		cost_ptr[i] = new double[sizeof(double) * n];

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			cost_ptr[i][j] = cost_c[i][j];
		}
	}

	int* x_c = new int[sizeof(int) * n];
	int *y_c = new int[sizeof(int) * n];

	int ret = lapjv_internal(n, cost_ptr, x_c, y_c);
	if (ret != 0)
	{
		cout << "Calculate Wrong!" << endl;
		system("pause");
		exit(0);
	}

	double opt = 0.0;

	if (n != n_rows)
	{
		for (int i = 0; i < n; i++)
		{
			if (x_c[i] >= n_cols)
				x_c[i] = -1;
			if (y_c[i] >= n_rows)
				y_c[i] = -1;
		}
		for (int i = 0; i < n_rows; i++)
		{
			rowsol[i] = x_c[i];
		}
		for (int i = 0; i < n_cols; i++)
		{
			colsol[i] = y_c[i];
		}

		if (return_cost)
		{
			for (int i = 0; i < rowsol.size(); i++)
			{
				if (rowsol[i] != -1)
				{
					// cout << i << "\t" << rowsol[i] << "\t" << cost_ptr[i][rowsol[i]] << endl;
					opt += cost_ptr[i][rowsol[i]];
				}
			}
		}
	}
	else if (return_cost)
	{
		for (int i = 0; i < rowsol.size(); i++)
		{
			opt += cost_ptr[i][rowsol[i]];
		}
	}
	#if TRACK_DEBUG
		// 匹配结果	
		std::cout << "当前匹配结果" << std::endl;
		for (int i = 0; i < rowsol.size(); i++)
		{
			std::cout << "(" << rowsol[i] << ")";
		}
		std::cout << std::endl;
		for (int i = 0; i < colsol.size(); i++)
		{
			std::cout << "(" << colsol[i] << ")";
		}
		std::cout << std::endl;
		std::cout << "最优成本 " << opt << std::endl;
	#endif

	for (int i = 0; i < n; i++)
	{
		delete[]cost_ptr[i];
	}
	delete[]cost_ptr;
	delete[]x_c;
	delete[]y_c;

	return opt; // 最优成本值
}

Scalar BYTETracker::get_color(int idx)
{
	idx += 3;
	return Scalar(37 * idx % 255, 17 * idx % 255, 29 * idx % 255);
}