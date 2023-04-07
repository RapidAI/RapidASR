#include "precomp.h"

bool CQmASRRecog::IsLoaded()
{

	return m_bIsLoaded;
}

CQmASRRecog::CQmASRRecog(const char* szModelDir,int nThread)
{
	
	m_bIsLoaded = LoadModel(szModelDir,nThread);
}

CQmASRRecog::CQmASRRecog(const char* szEncoder, const char* szDecoder, const char* szDict, const char* szConfig,int nThread)
{
	
	m_bIsLoaded = LoadModel(szEncoder, szDecoder, szDict, szConfig,nThread);
}
bool CQmASRRecog::LoadModel(const char* szModelDir,int nNumThread)
{

	string strEncoder, strDecoder;
	string strBaseDir= szModelDir;

	if (!szModelDir)
		return false;
	if (szModelDir[strlen(szModelDir) - 1] != OS_SEP[0])
	{
		strBaseDir = strBaseDir + OS_SEP;
	}

	strEncoder = strBaseDir + QM_ENCODER_MODEL;
	strDecoder = strBaseDir + QM_DECODER_MODEL;
	m_strDict = strBaseDir + QM_DICT_FILE;
	m_strConfig = strBaseDir + QM_CONFIG_FILE;



	return LoadModel(strEncoder.c_str(), strDecoder.c_str(), m_strDict.c_str(), m_strConfig.c_str(),nNumThread);
}

bool CQmASRRecog::LoadModel(const char* szEncoder, const char* szDecoder, const char* szDict, const char* szConfig, int nNumThread)
{

	sessionOptions.SetInterOpNumThreads(nNumThread);
	sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

	m_session_encoder = new Ort::Session(envEncoder, szEncoder, sessionOptions);
	m_session_decoder = new Ort::Session(envDecoder, szDecoder, sessionOptions);

	getInputNameAll(m_session_encoder, m_vecEncInputName);

	for (auto& item : m_vecEncInputName)
		m_strEncInputName.push_back(item.c_str());

	getOutputNameAll (m_session_encoder, m_vecEncOutputName);
	for (auto& item : m_vecEncOutputName)
		m_strEncOutputName.push_back(item.c_str());

	getInputNameAll (m_session_decoder, m_vecDecInputName);
	for (auto& item : m_vecDecInputName)
		m_strDecInputName.push_back(item.c_str());

	getOutputNameAll(m_session_decoder, m_vecDecOutputName);
	for (auto& item : m_vecDecOutputName)
		m_strDecOutputName.push_back(item.c_str());

	// load vocabulary

	ifstream fdict(szDict);
	if (!fdict.is_open())
		return false;

	char  strLine[101];
	string strToken;
	int nIndex;
	while(fdict.getline(strLine,100))
	{
		stringstream sstr;
		sstr.str(strLine);
		sstr >> strToken;
		sstr >> nIndex;
		m_Vocabulary.push_back(strToken);
	}

	// load config

	//model_conf:
	//	ctc_weight: 0.3
	//	length_normalized_loss : false
	//	lsm_weight : 0.1
	//	reverse_weight : 0.3
	try
	{
		YAML::Node conf = YAML::LoadFile(szConfig);
		auto var = conf["model_conf"]["reverse_weight"];

		try {
			m_reverse_weight = var.as<float>();
		}
		catch (YAML::TypedBadConversion<float>& e) {
			//std::cout << "label node is NULL" << std::endl;
			m_reverse_weight = 0.0f;
		}
	}
	catch (YAML::BadFile& e)
	{

		m_reverse_weight = 0.0f;
	}




	return true;
}

CQmASRRecog::~CQmASRRecog()
{


	if (m_session_encoder)
	{
		delete m_session_encoder;
		m_session_encoder = nullptr;
	}


	if (m_session_decoder)
	{
		delete m_session_decoder;
		m_session_decoder = nullptr;
	}

}


//
//  By default, it has a sample rate of 16bits.
//
//

// https://blog.csdn.net/hongmaodaxia/article/details/44224825


// http://fancyerii.github.io/kaldicodes/feature/

// https://github.com/kli017/wenet/tree/wenet-ort
int  CQmASRRecog::ExtractFeature(vector<float> & wav, std::vector<std::vector<float>>& feats, wenet::FeaturePipelineConfig& config)
{

	wenet::Fbank fbank_(config.num_bins,config.sample_rate,config.frame_length,config.frame_shift);
 //std::vector<float> waves;
 //waves.insert(waves.end(), wav.begin(), wav.end());
 // //waves.insert(waves.end(), remained_wav_.begin(), remained_wav_.end());
 //waves.insert(waves.end(), wav.begin(), wav.end());
 int num_frames = fbank_.Compute(wav, &feats);
  //for (size_t i = 0; i < feats.size(); ++i) {
	 // feature_queue_.Push(std::move(feats[i]));
  //}
 return num_frames;

  //int num_frames = fbank_.Compute(waves, &feats);
  //for (size_t i = 0; i < feats.size(); ++i) {
  //  feature_queue_.Push(std::move(feats[i]));
  //}
  //num_frames_ += num_frames;

  //int left_samples = waves.size() - config_.frame_shift * num_frames;
  //remained_wav_.resize(left_samples);
  //std::copy(waves.begin() + config_.frame_shift * num_frames, waves.end(),
  //          remained_wav_.begin());
  // We are still adding wave, notify input is not finished
  
}


PRAPIDASR_RECOG_RESULT CQmASRRecog::DoRecognize(vector<vector<float>> & feats, RAPIDASR_MODE Mode)
{

	PRAPIDASR_RECOG_RESULT pResult= new RAPIDASR_RECOG_RESULT;

	pResult->Result = QAC_ERROR;
	
	// for encoder model
	Ort::RunOptions run_option{nullptr};
	int num_frames = feats.size();
	
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

	std::vector<int64_t> input_mask_node_dims = { 1, num_frames, QM_FEATURE_DIMENSION };
	std::vector<float> flatfeats;
	for (auto& e : feats)
	{
		flatfeats.insert(flatfeats.end(), e.begin(), e.end());
	}

	Ort::Value onnx_feats =Ort::Value::CreateTensor<float>(memory_info,
		flatfeats.data(),
		flatfeats.size(),
		input_mask_node_dims.data(),
		input_mask_node_dims.size());

	std::vector<int32_t> feats_len{ num_frames };
	std::vector<int64_t> feats_len_dim{1};
	Ort::Value onnx_feats_len = Ort::Value::CreateTensor(
		memory_info,
		feats_len.data(),
		feats_len.size()*sizeof(int32_t),
		feats_len_dim.data(),
		feats_len_dim.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
	std::vector<Ort::Value> input_onnx;
	input_onnx.emplace_back(std::move(onnx_feats));
	input_onnx.emplace_back(std::move(onnx_feats_len));

	auto output= m_session_encoder->Run(run_option,
			m_strEncInputName.data(),
			input_onnx.data(),
			m_strEncInputName.size(),
			m_strEncOutputName.data(),
			m_strEncOutputName.size()
	);
	//    encoder_out, encoder_out_lens, ctc_log_probs, beam_log_probs, beam_log_probs_idx = output
	assert(output.size() == 5 && output[0].IsTensor());

	vector<int64_t> shape_encoder_out = output[0].GetTensorTypeAndShapeInfo().GetShape();
	auto encoder_out = output[0].GetTensorMutableData<float>();
	//int   nLen= std::accumulate(shape_encode_out.begin(), shape_encode_out.end(), 1, std::multiplies<int64_t>());
	vector<int64_t> shape_encoder_out_lens= output[1].GetTensorTypeAndShapeInfo().GetShape();
	auto encoder_out_lens = output[1].GetTensorMutableData<int32>();

	vector<int64_t> shape_ctc_log_probs = output[2].GetTensorTypeAndShapeInfo().GetShape();
	auto ctc_log_probs = output[2].GetTensorMutableData<float>();

	vector<int64_t> shape_beam_log_probs = output[3].GetTensorTypeAndShapeInfo().GetShape();
	auto beam_log_probs = output[3].GetTensorMutableData<float>();

	vector<int64_t> shape_beam_log_probs_idx = output[4].GetTensorTypeAndShapeInfo().GetShape();
	auto beam_log_probs_idx = output[4].GetTensorMutableData<int64>();

	auto beam_size = shape_beam_log_probs[2];
	auto batch_size = shape_beam_log_probs[0];

	int num_process = 2; // the number of processors.

	int sos, eos;
	sos = eos = m_Vocabulary.size() -1;

	if (Mode == RPASRM_CTC_GREEDY_SEARCH) //ctc greedy search
	{
		if (beam_size != 1)
		{
			vector<int> log_probs_idx;
			for (int i = 0; i < shape_beam_log_probs_idx[1]; i++)
			{
				//log_probs_idx = beam_log_probs_idx[:, : , 0]
				log_probs_idx.push_back(*(beam_log_probs_idx+shape_beam_log_probs_idx[2]*i));
			}

			vector<vector<int>> batch_sents;
			batch_sents.push_back(log_probs_idx);
			auto sentence = map_batch(batch_sents, m_Vocabulary, num_process,true,0);

			pResult->Strings = sentence;
		
			
		}
	}
	else
	if (Mode == RPASRM_CTC_RPEFIX_BEAM_SEARCH || Mode == RPASRM_ATTENSION_RESCORING)
	{
		vector<vector<vector<double>>> batch_log_probs_seq;
		vector<vector<vector<int>>> batch_log_probs_idx;
		vector<PathTrie*> batch_root;
		vector<bool>  batch_start;
		size_t beam_size= shape_beam_log_probs[2];
		size_t batch_size= shape_beam_log_probs[0];;

		for (int i = 0; i < shape_encoder_out_lens[0]; i++)
		{
			auto num_sent = encoder_out_lens[i];

			vector <vector<double>> batch_log_probs_seq_list;
			vector <vector<int>> batch_log_probs_index_list;

			for (int s = 0; s < num_sent; s++)
			{
				vector<double> temp;
				for (int t = 0; t < shape_beam_log_probs[2]; t++)
					temp.push_back(beam_log_probs[s * shape_beam_log_probs[2] + t]);
				batch_log_probs_seq_list.push_back(temp);

				vector<int> tempindex;

				for (int t = 0; t < shape_beam_log_probs_idx[2]; t++)
					tempindex.push_back(beam_log_probs_idx[s * shape_beam_log_probs_idx[2] + t]);

				batch_log_probs_index_list.push_back(tempindex);
			}

			batch_root.push_back(new PathTrie);

			batch_log_probs_seq.push_back(batch_log_probs_seq_list);

			batch_log_probs_idx.push_back(batch_log_probs_index_list);
			batch_start.push_back(true);
		}

		auto score_hyps=ctc_beam_search_decoder_batch(batch_log_probs_seq, batch_log_probs_idx, batch_root, batch_start, beam_size,num_process,0, -2, 0.99999);
		if (Mode == RPASRM_CTC_RPEFIX_BEAM_SEARCH)
		{
			vector<std::vector<int>> batch_sents;
			for (auto& item :score_hyps)
			{
				
				
				batch_sents.push_back( item[0].second);
			}
			auto sentences = map_batch(batch_sents, m_Vocabulary, num_process, false, 0);
		
			pResult->Strings = sentences;

		}

		if (Mode == RPASRM_ATTENSION_RESCORING)
		{
			int max_len = 0;
			vector<vector<float>> ctc_score;
			vector<vector<int>> all_hyps;
			for (auto& hyps : score_hyps)
			{
				auto cur_len = hyps.size();

				if ( cur_len < beam_size)
				{

					vector<int> tmp;
					for (int s = 0; s < hyps[0].second.size(); s++)
						tmp.push_back(0);

					for (int i = 0; i< beam_size - cur_len; i++)
					{
					
						hyps.push_back(std::make_pair(-999999999999, tmp));
					}

					 // hyps += (beam_size - cur_len) * [(-float("INF"), (0, ))]
				}
				vector<float> cur_ctc_score;
			
				
				for (auto& hyp : hyps)
				{
					cur_ctc_score.push_back(hyp.first);
					all_hyps.push_back(hyp.second);
					if (hyp.second.size() > max_len)
						max_len = hyp.second.size();
				}
				ctc_score.push_back(cur_ctc_score);
			}

			// hyps_pad_sos_eos
			//  r_hyps_pad_sos_eos

			auto lastdim = max_len + 2;
			auto eos_len = batch_size * beam_size * lastdim;
			vector<int64_t> hyps_pad_sos_eos(eos_len, IGNORE_ID);
			vector<int64_t> r_hyps_pad_sos_eos(eos_len, IGNORE_ID);
			vector<int32_t> hyps_lens_sos(batch_size* beam_size,1);
			int k = 0;
			for (int i = 0; i < batch_size; i++)
			{
				for (int j = 0; j < beam_size; j++)
				{
					vector<int64_t> tmp,rtmp;
					auto cand = all_hyps[k];
					auto rcand = cand;
					reverse(rcand.begin(), rcand.end());
					int l = cand.size() + 2;
					tmp.push_back(sos);
					rtmp.push_back(sos);
					tmp.insert(tmp.begin()+1,cand.begin(), cand.end());
					rtmp.insert(rtmp.begin()+1,rcand.begin(), rcand.end());
					tmp.push_back(eos);
					rtmp.push_back(eos);
					copy( tmp.begin(), tmp.end(), hyps_pad_sos_eos.begin() + i * j * lastdim);
					copy( rtmp.begin(), rtmp.end(), r_hyps_pad_sos_eos.begin() + i * j * lastdim);
					hyps_lens_sos[beam_size * i + j] = cand.size() + 1;
					k++;
				}
			}
			
			Ort::Value onnx_encoder_out = Ort::Value::CreateTensor<float>(memory_info,
				encoder_out,
				accumulate(shape_encoder_out.begin(), shape_encoder_out.end(), 1, multiplies<int>()),
				shape_encoder_out.data(),
				shape_encoder_out.size()
				);

			Ort::Value onnx_encoder_out_len = Ort::Value::CreateTensor<int32_t>(memory_info,
				encoder_out_lens,
				accumulate(shape_encoder_out_lens.begin(), shape_encoder_out_lens.end(), 1, multiplies<int>()),
				shape_encoder_out_lens.data(),
				shape_encoder_out_lens.size());
	
	
			std::vector<int64_t> hyps_pad_sos_eos_dims = { batch_size, beam_size, lastdim };
			Ort::Value onnx_hyps_pad_sos_eos = Ort::Value::CreateTensor<int64_t>(memory_info,
				hyps_pad_sos_eos.data(),
				hyps_pad_sos_eos.size(),
				hyps_pad_sos_eos_dims.data(),
				hyps_pad_sos_eos_dims.size());

			//hyps_pad_sos_eos

			std::vector<int64_t> r_hyps_pad_sos_eos_dims = { batch_size, beam_size, lastdim };
			Ort::Value onnx_r_hyps_pad_sos_eos = Ort::Value::CreateTensor<int64_t>(memory_info,
				r_hyps_pad_sos_eos.data(),
				r_hyps_pad_sos_eos.size(),
				r_hyps_pad_sos_eos_dims.data(),
				r_hyps_pad_sos_eos_dims.size());

			

			std::vector<int64_t> hyps_len_sos_dims = { batch_size, beam_size };
			Ort::Value onnx_hyps_len_sos = Ort::Value::CreateTensor<int32_t>(memory_info,
				hyps_lens_sos.data(),
				hyps_lens_sos.size(),
				hyps_len_sos_dims.data(),
				hyps_len_sos_dims.size());



			std::vector<float> flat_ctc_score;
			for (auto& e : ctc_score)
			{
				flat_ctc_score.insert(flat_ctc_score.end(), e.begin(), e.end());
			}
			std::vector<int64_t> ctc_score_dims = { batch_size, beam_size };
			Ort::Value onnx_ctc_score = Ort::Value::CreateTensor<float>(memory_info,
				flat_ctc_score.data(),
				flat_ctc_score.size(),
				ctc_score_dims.data(),
				ctc_score_dims.size());

			std::vector<Ort::Value> input_onnx;
			input_onnx.emplace_back(std::move(onnx_encoder_out));
			input_onnx.emplace_back(std::move(onnx_encoder_out_len));
			input_onnx.emplace_back(std::move(onnx_hyps_pad_sos_eos));
			input_onnx.emplace_back(std::move(onnx_hyps_len_sos));
			if(m_reverse_weight)
				input_onnx.emplace_back(std::move(onnx_r_hyps_pad_sos_eos));
			else
				input_onnx.emplace_back(std::move(onnx_ctc_score));
				
				

			auto decoder_output = m_session_decoder->Run(run_option,
				m_strDecInputName.data(),
				input_onnx.data(),
				m_strDecInputName.size(),
				m_strDecOutputName.data(),
				m_strDecOutputName.size()
			);

			assert(decoder_output.size() == 1 && decoder_output[0].IsTensor());
			//auto best_index = decoder_output[0];
			vector<int64_t> shape_best_index = decoder_output[0].GetTensorTypeAndShapeInfo().GetShape();
			auto best_index = decoder_output[0].GetTensorMutableData<int64_t>(); 
			k = 0;
			vector<vector<int>>  batch_sents;
			for (int i=0; i< shape_best_index[0]; i++)
			{
				if (best_index[i] >= (beam_size - k)) // 如果index 大于  选出的个数，则非法。
					return nullptr;
				batch_sents.push_back(all_hyps[k+ best_index[i]]);
				k += beam_size;
			}

			auto sentences = map_batch(batch_sents, m_Vocabulary, num_process);

			pResult->Strings = sentences;
			
	
		}

		for (auto& item : batch_root)
			delete item;

		

	}

	pResult->Result = QAC_OK;
	return pResult;
}
