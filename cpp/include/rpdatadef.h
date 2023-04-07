#pragma once


#define  QM_ENCODER_MODEL  "encoder.onnx"
#define  QM_DECODER_MODEL  "decoder.onnx"
#define  QM_CONFIG_FILE		"train.yaml"
#define  QM_DICT_FILE		"words.txt"
#define  IGNORE_ID			-1

#define	QM_FEATURE_DIMENSION  80
#define QM_DEFAULT_SAMPLE_RATE  16000

#ifdef WIN32

#define OS_SEP  "\\"
#else
#define OS_SEP  "/"
#endif
typedef struct
{
	void* p;
} RAPIDASR_CONTEXT,*PRAPIDASR_CONTEXT;


typedef enum
{
QAC_ERROR=-1,
QAC_OK=0,

}RAPIDASR_CODE;
typedef struct
{
	RAPIDASR_CODE Result;
	vector<string> Strings;

} RAPIDASR_RECOG_RESULT,*PRAPIDASR_RECOG_RESULT;