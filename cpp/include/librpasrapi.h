#pragma once


#ifdef WIN32


#ifdef D_RAPIDASR_API_EXPORT

#define  _RAPIDASRAPI __declspec(dllexport)
#else
#define  _RAPIDASRAPI __declspec(dllimport)
#endif
	

#else
#define _RAPIDASRAPI  
#endif


#ifdef __cplusplus 

extern "C" {
#endif

typedef void* RAPIDASR_HANDLE;

typedef void* RAPIDASR_RESULT;

typedef unsigned char RAPIDASR_BOOL;

#define RAPIDASR_TRUE 1
#define RAPIDASR_FALSE 0
#define RP_DEFAULT_THREAD_NUM  4


typedef enum
{
 RPASRM_CTC_GREEDY_SEARCH=0,
 RPASRM_CTC_RPEFIX_BEAM_SEARCH = 1,
 RPASRM_ATTENSION_RESCORING = 2,
 
}RAPIDASR_MODE;
	
	// APIs for qmasr

_RAPIDASRAPI RAPIDASR_HANDLE RpASR_init(const char* szModelDir, int nThread);


_RAPIDASRAPI RAPIDASR_RESULT RpASRRecogBuffer(RAPIDASR_HANDLE handle, const char* szBuf, int nLen, RAPIDASR_MODE Mode);
_RAPIDASRAPI RAPIDASR_RESULT RpASRRecogFile(RAPIDASR_HANDLE handle, const char* szWavfile, RAPIDASR_MODE Mode);

_RAPIDASRAPI const char* RpASRGetResult(RAPIDASR_RESULT Result,int nIndex);

_RAPIDASRAPI const int RpASRGetRetNumber(RAPIDASR_RESULT Result);
_RAPIDASRAPI void RpASRFreeResult(RAPIDASR_RESULT Result);


_RAPIDASRAPI void RpASR_Uninit(RAPIDASR_HANDLE Handle);





#ifdef __cplusplus 

}
#endif