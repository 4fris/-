#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <signal.h>
#include <pthread.h>
#include <sys/prctl.h>
#include <math.h>
#include "config.h"
#include "hi_common.h"
#include "hi_comm_sys.h"
#include "hi_comm_svp.h"
#include "sample_comm.h"
#include "sample_comm_svp.h"
#include "sample_comm_nnie.h"
#include "nnie_face_api.h"
#include "sample_svp_nnie_software.h"
#include "sample_comm_ive.h"
#include "hi_type.h"

//#include"list.h"

/*cnn para*/
static SAMPLE_SVP_NNIE_MODEL_S s_stDetModel = {0};
static SAMPLE_SVP_NNIE_MODEL_S s_stExtModel = {0};
static SAMPLE_SVP_NNIE_PARAM_S s_stDetNnieParam = {0};
static SAMPLE_SVP_NNIE_PARAM_S s_stExtNnieParam = {0};
//static anchor_generator_t* anc_gen = NULL;
int IsDebugLog = 0;
SAMPLE_SVP_NNIE_CFG_S   stNnieCfg = {0};
HI_S32 as32ResultDet[200 * 15] = { 0 };
HI_S32 u32ResultDetCnt = 0;
int IndexBuffer[512] = { 0 };
#ifdef SAMPLE_SVP_NNIE_PERF_STAT
#define SAMPLE_SVP_NNIE_PERF_STAT_OP_FORWARD_CLREAR()  memset(&s_stOpForwardPerfTmp,0,sizeof(s_stOpForwardPerfTmp));
#define SAMPLE_SVP_NNIE_PERF_STAT_OP_FORWARD_SRC_FLUSH_TIME() SAMPLE_SVP_NNIE_PERF_STAT_GET_DIFF_TIME(s_stOpForwardPerfTmp.u64SrcFlushTime)
#define SAMPLE_SVP_NNIE_PERF_STAT_OP_FORWARD_PRE_DST_FLUSH_TIME() SAMPLE_SVP_NNIE_PERF_STAT_GET_DIFF_TIME(s_stOpForwardPerfTmp.u64PreDstFulshTime)
#define SAMPLE_SVP_NNIE_PERF_STAT_OP_FORWARD_AFTER_DST_FLUSH_TIME() SAMPLE_SVP_NNIE_PERF_STAT_GET_DIFF_TIME(s_stOpForwardPerfTmp.u64AferDstFulshTime)
#define SAMPLE_SVP_NNIE_PERF_STAT_OP_FORWARD_OP_TIME() SAMPLE_SVP_NNIE_PERF_STAT_GET_DIFF_TIME(s_stOpForwardPerfTmp.u64OPTime)


static SAMPLE_SVP_NNIE_OP_PERF_STAT_S   s_stOpForwardPerfTmp = {0};
#else

#define SAMPLE_SVP_NNIE_PERF_STAT_OP_FORWARD_CLREAR()
#define SAMPLE_SVP_NNIE_PERF_STAT_OP_FORWARD_SRC_FLUSH_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_OP_FORWARD_PRE_DST_FLUSH_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_OP_FORWARD_AFTER_DST_FLUSH_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_OP_FORWARD_OP_TIME()

#endif

float sigmoid(float x){
	return (1.0f / ((float)exp((double)(-x)) + 1.0f));
}

/******************************************************************************
* function : NNIE Forward
******************************************************************************/
static HI_S32 SAMPLE_SVP_NNIE_Forward(SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
    SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S* pstInputDataIdx,
    SAMPLE_SVP_NNIE_PROCESS_SEG_INDEX_S* pstProcSegIdx,HI_BOOL bInstant)
{
    HI_S32 s32Ret = HI_SUCCESS;
    HI_U32 i = 0, j = 0;
    HI_BOOL bFinish = HI_FALSE;
    SVP_NNIE_HANDLE hSvpNnieHandle = 0;
    HI_U32 u32TotalStepNum = 0;
    SAMPLE_SVP_NIE_PERF_STAT_DEF_VAR();

    SAMPLE_SVP_NNIE_PERF_STAT_OP_FORWARD_CLREAR();

    SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx].stTskBuf.u64PhyAddr,
        SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID,pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx].stTskBuf.u64VirAddr),
        pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx].stTskBuf.u32Size);

    SAMPLE_SVP_NNIE_PERF_STAT_BEGIN();
    for(i = 0; i < pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx].u32DstNum; i++)
    {
        if(SVP_BLOB_TYPE_SEQ_S32 == pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].enType)
        {
            for(j = 0; j < pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Num; j++)
            {
                u32TotalStepNum += *(SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_U32,pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stSeq.u64VirAddrStep)+j);
            }
            SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64PhyAddr,
                SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID,pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64VirAddr),
                u32TotalStepNum*pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Stride);

        }
        else
        {
            SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64PhyAddr,
                SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID,pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64VirAddr),
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Num*
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stWhc.u32Chn*
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stWhc.u32Height*
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Stride);
        }
    }
    SAMPLE_SVP_NNIE_PERF_STAT_END();
    SAMPLE_SVP_NNIE_PERF_STAT_OP_FORWARD_PRE_DST_FLUSH_TIME();

    /*set input blob according to node name*/
    if(pstInputDataIdx->u32SegIdx != pstProcSegIdx->u32SegIdx)
    {
        for(i = 0; i < pstNnieParam->pstModel->astSeg[pstProcSegIdx->u32SegIdx].u16SrcNum; i++)
        {
            for(j = 0; j < pstNnieParam->pstModel->astSeg[pstInputDataIdx->u32SegIdx].u16DstNum; j++)
            {
                if(0 == strncmp(pstNnieParam->pstModel->astSeg[pstInputDataIdx->u32SegIdx].astDstNode[j].szName,
                    pstNnieParam->pstModel->astSeg[pstProcSegIdx->u32SegIdx].astSrcNode[i].szName,
                    SVP_NNIE_NODE_NAME_LEN))
                {
                    pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astSrc[i] =
                        pstNnieParam->astSegData[pstInputDataIdx->u32SegIdx].astDst[j];
                    break;
                }
            }
            SAMPLE_SVP_CHECK_EXPR_RET((j == pstNnieParam->pstModel->astSeg[pstInputDataIdx->u32SegIdx].u16DstNum),
                HI_FAILURE,SAMPLE_SVP_ERR_LEVEL_ERROR,"Error,can't find %d-th seg's %d-th src blob!\n",
                pstProcSegIdx->u32SegIdx,i);
        }
    }

    /*NNIE_Forward*/
    SAMPLE_SVP_NNIE_PERF_STAT_BEGIN();
    s32Ret = HI_MPI_SVP_NNIE_Forward(&hSvpNnieHandle, // 官方api, 将图片输入NNIE做前向处理 。
        pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astSrc,
        pstNnieParam->pstModel, pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst, // astDst 包含 最终结果输出 以及 中间层输出（需要设置上报） 。
        &pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx], bInstant);
    SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret,s32Ret,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,HI_MPI_SVP_NNIE_Forward failed!\n");

    if(bInstant) // 该值为 HI_MPI_SVP_NNIE_Forward 输出结果标志 。
    {
        /*Wait NNIE finish*/
        while(HI_ERR_SVP_NNIE_QUERY_TIMEOUT == (s32Ret = HI_MPI_SVP_NNIE_Query(pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx].enNnieId,
            hSvpNnieHandle, &bFinish, HI_TRUE)))
        {
            usleep(100);
            SAMPLE_SVP_TRACE(SAMPLE_SVP_ERR_LEVEL_INFO,
                "HI_MPI_SVP_NNIE_Query Query timeout!\n");
        }
    }
    SAMPLE_SVP_NNIE_PERF_STAT_END();
    SAMPLE_SVP_NNIE_PERF_STAT_OP_FORWARD_OP_TIME();
    u32TotalStepNum = 0;

    SAMPLE_SVP_NNIE_PERF_STAT_BEGIN();
    for(i = 0; i < pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx].u32DstNum; i++)
    {
        if(SVP_BLOB_TYPE_SEQ_S32 == pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].enType)
        {
            for(j = 0; j < pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Num; j++)
            {
                u32TotalStepNum += *(SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_U32,pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stSeq.u64VirAddrStep)+j);
            }
            SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64PhyAddr,
                SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID,pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64VirAddr),
                u32TotalStepNum*pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Stride);

        }
        else
        {
            SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64PhyAddr,
                SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID,pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64VirAddr),
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Num*
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stWhc.u32Chn*
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stWhc.u32Height*
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Stride);
        }
    }
    SAMPLE_SVP_NNIE_PERF_STAT_END();
    SAMPLE_SVP_NNIE_PERF_STAT_OP_FORWARD_AFTER_DST_FLUSH_TIME();

    return s32Ret;
}

/******************************************************************************
* function : Fill Src Data
******************************************************************************/
static HI_S32 SAMPLE_SVP_NNIE_FillSrcData(SAMPLE_SVP_NNIE_CFG_S* pstNnieCfg,
    SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam, SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S* pstInputDataIdx)
{
    FILE* fp = NULL;
    HI_U32 i =0, j = 0, n = 0;
    HI_U32 u32Height = 0, u32Width = 0, u32Chn = 0, u32Stride = 0, u32Dim = 0;
    HI_U32 u32VarSize = 0;
    HI_S32 s32Ret = HI_SUCCESS;
    HI_U8*pu8PicAddr = NULL;
    HI_U32*pu32StepAddr = NULL;
    HI_U32 u32SegIdx = pstInputDataIdx->u32SegIdx;
    HI_U32 u32NodeIdx = pstInputDataIdx->u32NodeIdx;
    HI_U32 u32TotalStepNum = 0;
    //printf("Info, open file!\n");
    /*open file*/
    if (NULL != pstNnieCfg->pszPic) // pszPic 是 bgr 图片路径
    {
        fp = fopen(pstNnieCfg->pszPic,"rb");
        SAMPLE_SVP_CHECK_EXPR_RET(NULL == fp,HI_INVALID_VALUE,SAMPLE_SVP_ERR_LEVEL_ERROR,
            "Error, open file failed!\n");
    }

    /*get data size*/
    if(SVP_BLOB_TYPE_U8 <= pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].enType &&
        SVP_BLOB_TYPE_YVU422SP >= pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].enType)
    {
        u32VarSize = sizeof(HI_U8);
    }
    else
    {
        u32VarSize = sizeof(HI_U32);
    }

    /*fill src data*/
    if(SVP_BLOB_TYPE_SEQ_S32 == pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].enType)
    {
        u32Dim = pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].unShape.stSeq.u32Dim;
        u32Stride = pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u32Stride;
        pu32StepAddr = SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_U32,pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].unShape.stSeq.u64VirAddrStep);
        pu8PicAddr = SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_U8,pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u64VirAddr);
        for(n = 0; n < pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u32Num; n++)
        {
            for(i = 0;i < *(pu32StepAddr+n); i++)
            {
                s32Ret = fread(pu8PicAddr,u32Dim*u32VarSize,1,fp);
                SAMPLE_SVP_CHECK_EXPR_GOTO(1 != s32Ret,FAIL,SAMPLE_SVP_ERR_LEVEL_ERROR,"Error,Read image file failed!\n");
                pu8PicAddr += u32Stride;
            }
            u32TotalStepNum += *(pu32StepAddr+n);
        }
        SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u64PhyAddr,
            SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID,pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u64VirAddr),
            u32TotalStepNum*u32Stride);
    }
    else
    {
        u32Height = pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].unShape.stWhc.u32Height;
        u32Width = pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].unShape.stWhc.u32Width;
        u32Chn = pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].unShape.stWhc.u32Chn;
        u32Stride = pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u32Stride;
        pu8PicAddr = SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_U8,pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u64VirAddr);
        if(SVP_BLOB_TYPE_YVU420SP== pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].enType)
        {
            for(n = 0; n < pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u32Num; n++)
            {
                for(i = 0; i < u32Chn*u32Height/2; i++)
                {
                    s32Ret = fread(pu8PicAddr,u32Width*u32VarSize,1,fp);
                    SAMPLE_SVP_CHECK_EXPR_GOTO(1 != s32Ret,FAIL,SAMPLE_SVP_ERR_LEVEL_ERROR,"Error,Read image file failed!\n");
                    pu8PicAddr += u32Stride;
                }
            }
        }
        else if(SVP_BLOB_TYPE_YVU422SP== pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].enType)
        {
            for(n = 0; n < pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u32Num; n++)
            {
                for(i = 0; i < u32Height*2; i++)
                {
                    s32Ret = fread(pu8PicAddr,u32Width*u32VarSize,1,fp);
                    SAMPLE_SVP_CHECK_EXPR_GOTO(1 != s32Ret,FAIL,SAMPLE_SVP_ERR_LEVEL_ERROR,"Error,Read image file failed!\n");
                    pu8PicAddr += u32Stride;
                }
            }
        }
        else
        {
            for(n = 0; n < pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u32Num; n++)
            {
                for(i = 0;i < u32Chn; i++)
                {
                    for(j = 0; j < u32Height; j++)
                    {
                        s32Ret = fread(pu8PicAddr,u32Width*u32VarSize,1,fp);
                        SAMPLE_SVP_CHECK_EXPR_GOTO(1 != s32Ret,FAIL,SAMPLE_SVP_ERR_LEVEL_ERROR,"Error,Read image file failed!\n");
                        pu8PicAddr += u32Stride;
                    }
                }
            }
        }
        SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u64PhyAddr,
            SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID,pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u64VirAddr),
            pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u32Num*u32Chn*u32Height*u32Stride);
    }

    fclose(fp);
    return HI_SUCCESS;
FAIL:

    fclose(fp);
    return HI_FAILURE;
}

/******************************************************************************
* function : Cnn Deinit
******************************************************************************/
static HI_S32 SAMPLE_SVP_NNIE_Cnn_Deinit(SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam, SAMPLE_SVP_NNIE_MODEL_S* pstNnieModel)
{

    HI_S32 s32Ret = HI_SUCCESS;
    /*hardware para deinit*/
    if(pstNnieParam!=NULL)
    {
        s32Ret = SAMPLE_COMM_SVP_NNIE_ParamDeinit(pstNnieParam);
        SAMPLE_SVP_CHECK_EXPR_TRACE(HI_SUCCESS != s32Ret,SAMPLE_SVP_ERR_LEVEL_ERROR,
            "Error,SAMPLE_COMM_SVP_NNIE_ParamDeinit failed!\n");
    }
    /*model deinit*/
    if(pstNnieModel!=NULL)
    {
        s32Ret = SAMPLE_COMM_SVP_NNIE_UnloadModel(pstNnieModel);
        SAMPLE_SVP_CHECK_EXPR_TRACE(HI_SUCCESS != s32Ret,SAMPLE_SVP_ERR_LEVEL_ERROR,
            "Error,SAMPLE_COMM_SVP_NNIE_UnloadModel failed!\n");
    }
    return s32Ret;
}

/******************************************************************************
* function : Cnn init
******************************************************************************/
static HI_S32 SAMPLE_SVP_NNIE_Cnn_ParamInit(SAMPLE_SVP_NNIE_CFG_S* pstNnieCfg,
    SAMPLE_SVP_NNIE_PARAM_S *pstCnnPara)
{
    HI_S32 s32Ret = HI_SUCCESS;
    /*init hardware para*/
    s32Ret = SAMPLE_COMM_SVP_NNIE_ParamInit(pstNnieCfg,pstCnnPara); // 初始化 NNIE 硬件层参数 。
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret,INIT_FAIL_0,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error(%#x),SAMPLE_COMM_SVP_NNIE_ParamInit failed!\n",s32Ret);

    return s32Ret;
INIT_FAIL_0:
    s32Ret = SAMPLE_SVP_NNIE_Cnn_Deinit(pstCnnPara, NULL); // 如果失败，反初始化 。
    SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret,s32Ret,SAMPLE_SVP_ERR_LEVEL_ERROR,
            "Error(%#x),SAMPLE_SVP_NNIE_Cnn_Deinit failed!\n",s32Ret);
    return HI_FAILURE;

}

static unsigned int yolo_result_process(SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam, float *strides, anchor_w_h (*anchor_grids)[3], int *map_size, yolo_result **output_result, float confidence_threshold)
{
    HI_S32 output_num = 0;
	
	HI_S32 anchor_num = 0;
	
	HI_S32 feature_length = 0;

	float anchor_w = 0.0f;

	float anchor_h = 0.0f;

	int x = 0;
	
	int y = 0;

	HI_S32* output_addr = NULL;

	float confidence = 0.0f;

	float class_confidence = 0.0f;

	//float confidence_threshold = 0.4f;

	float pred_x = 0.0f;

	float pred_y = 0.0f;

	float pred_w = 0.0f;

	float pred_h = 0.0f;

	yolo_result *current = NULL;
	
	yolo_result *former = NULL;
	
	*output_result = NULL;

	unsigned int resltu_num = 0;
	// 取数据这里，我写篇博客说明一下，之前卡了我一会，觉得需要记录一下，帮助自己也帮助他人
    for (int yolo_layer_index = 0; yolo_layer_index < yolo_layer_num; yolo_layer_index ++) { // 3 yolo layer 
    
		feature_length = pstNnieParam->astSegData[0].astDst[yolo_layer_index].unShape.stWhc.u32Width; // 1600 / 400 / 100
		
		output_num = pstNnieParam->astSegData[0].astDst[yolo_layer_index].unShape.stWhc.u32Height; // 53
		
		anchor_num = pstNnieParam->astSegData[0].astDst[yolo_layer_index].unShape.stWhc.u32Chn; // 3
		
		output_addr = (HI_S32* )((HI_U8* )pstNnieParam->astSegData[0].astDst[yolo_layer_index].u64VirAddr); // yolo 输出的首地址
		
		for (int anchor_index = 0; anchor_index < anchor_num; anchor_index ++){ // 每个 grid 上有三个 anchor
				
				anchor_w = anchor_grids[yolo_layer_index][anchor_index].anchor_w;
				anchor_h = anchor_grids[yolo_layer_index][anchor_index].anchor_h;
				
				for (int coord_x_y = 0; coord_x_y < feature_length; coord_x_y ++){ // feature size 拉直后的长度，如 1600 400 100.
					y = coord_x_y / map_size[yolo_layer_index];
					x = coord_x_y % map_size[yolo_layer_index];
					
					confidence = *(output_addr + anchor_index * feature_length * output_num + 4 * feature_length + coord_x_y) / 4096.0f;  // confidence
					confidence = sigmoid(confidence);
					
					if (confidence > confidence_threshold){
						
						for (int output_index = 5; output_index < output_num; output_index ++){
							class_confidence = *(output_addr + anchor_index * feature_length * output_num + output_index * feature_length + coord_x_y) / 4096.0f;  // class confidence
							class_confidence = sigmoid(class_confidence) * confidence;
							// 注意，yolo v5 的类别置信度并不需要选出个最大值，它的 label 是多标签，所以并不是 softmax，我在博客里说明一下
							if (class_confidence > confidence_threshold){
								
								pred_x = *(output_addr + anchor_index * feature_length * output_num + 0 * feature_length + coord_x_y) / 4096.0f; // x
								pred_y = *(output_addr + anchor_index * feature_length * output_num + 1 * feature_length + coord_x_y) / 4096.0f; // y
								pred_w = *(output_addr + anchor_index * feature_length * output_num + 2 * feature_length + coord_x_y) / 4096.0f; // w
								pred_h = *(output_addr + anchor_index * feature_length * output_num + 3 * feature_length + coord_x_y) / 4096.0f; // h 

								pred_x = sigmoid(pred_x);
								pred_y = sigmoid(pred_y);
								pred_w = sigmoid(pred_w);
								pred_h = sigmoid(pred_h);
								// bbox 输出结果处理
								pred_x = (pred_x * 2.0f - 0.5f + (float)x) * strides[yolo_layer_index];
								pred_y = (pred_y * 2.0f - 0.5f + (float)y) * strides[yolo_layer_index];
								pred_w = (pred_w * 2.0f) * (pred_w * 2.0f) * anchor_w;
								pred_h = (pred_h * 2.0f) * (pred_h * 2.0f) * anchor_h;
								
								current = (yolo_result *) malloc(sizeof(yolo_result));
								// 坐标转换 (x y w h) -> (x y x y)
								current->left_up_x = YOLO_MAX((pred_x - 0.5f * (pred_w - 1.0f)), 0.0f);
								current->left_up_y = YOLO_MAX((pred_y - 0.5f * (pred_h - 1.0f)), 0.0f);
								current->right_down_x = YOLO_MIN((pred_x + 0.5f * (pred_w - 1.0f)), IMAGE_W);
								current->right_down_y = YOLO_MIN((pred_y + 0.5f * (pred_h - 1.0f)), IMAGE_H);
								
								current->class_index = output_index - 5; // 类别索引，减 5 是因为前五个数据是 bbox + confidence 输出.
								current->score = class_confidence; // 置信度
								current->next = NULL;
								resltu_num ++;
								
								if (*output_result == NULL){ // 存储结果
									*output_result = current;
									former = current;
								
								}else{
									former->next = current;
									former = former->next;
								}
								current = NULL;
							}
						}
					}
				}
		}
    }
	return resltu_num;
}
 
/*当第一个结构体位置调换时， output会被换到后面节点，使得其前面个别节点会丢失，除非传入双重指针 或者 链表设计头节点 */
void yolo_result_sort_test(yolo_result *output_result){ // 不可用，未写完
	
	yolo_result *comparable_node = NULL; // 右节点，挨个指向右边所有节点
	
	yolo_result *comparable_former_node = NULL;
	
	yolo_result *comparable_next_node = NULL;
	
	yolo_result *current_node = output_result; // 左节点，其与右边每个节点做比较

	yolo_result *current_former_node = NULL;

	yolo_result *current_next_node = NULL;
	
	yolo_result *temp_node = NULL;
	
	while (current_node != NULL){
		comparable_former_node = current_node;
		
		comparable_node = current_node->next;
		
		while (comparable_node != NULL){
			
			printf("current_node->score = %f\n", current_node->score);
		
			if (current_node->score >= comparable_node->score){ // 如果大于它，说明后面的比它小，比较下一个
				printf("1. comparable_node->score = %f\n", comparable_node->score);
				
				comparable_former_node = comparable_node;
				
				comparable_node = comparable_node->next;
				
			}else{
				// 当大于 current_confidence 时，调换位置，小的放后面去
				printf("2. comparable_node->score = %f\n", comparable_node->score);
				if (current_node->next == comparable_node){ // 如果二者是前后连接的状态
					 
					current_next_node = current_node; // 因为 current_node 要换到后面去，所以这样
					
					comparable_former_node = comparable_node; // comparable_node 等下会被换到前面
					
				}else{
					current_next_node = current_node->next;
				}
				comparable_next_node = comparable_node->next;
				
				temp_node = current_node;
				
				current_node = comparable_node;
				
				comparable_node = temp_node;
				printf("3. comparable_node->score = %f\n", comparable_node->score);
				if (current_former_node != NULL){ // 说明左边节点还在首节点位置
					
					current_former_node->next = current_node; // 接好链表
				}
				
				current_node->next = current_next_node;
				
				comparable_former_node->next = comparable_node; // 接好链表
				
				comparable_node->next = comparable_next_node; // 接好链表

				comparable_former_node = comparable_node; //更新位置，因为当前节点小于current_node ，不必再做比较

				comparable_node = comparable_node->next;
				
			}
			
		}
		printf("end one loop \n");
		current_former_node = current_node;
		current_node = current_node->next;
	}
}

void yolo_result_sort(yolo_result *output_result){ // 目前用这个做排序
	
	yolo_result *comparable_node = NULL; // 右节点，挨个指向右边所有节点
	
	yolo_result *comparable_next_node = NULL;
	
	yolo_result *current_node = output_result; // 左节点，其与右边每个节点做比较

	yolo_result *current_next_node = NULL;
	
	yolo_result temp_node = {0};
	
	while (current_node != NULL){
	
		comparable_node = current_node->next;
	
		current_next_node = current_node->next; // 记录后续节点，方便调换数据后维持链表完整
	
		while (comparable_node != NULL){
			
			comparable_next_node = comparable_node->next; // 记录后续节点，方便调换数据后维持链表完整
			
			if (current_node->score >= comparable_node->score){ // 如果大于它，说明后面的比它小，比较下一个
				
				comparable_node = comparable_node->next;
				
			}else{
				// 当大于 current_confidence 时，数据做调换，内存不变，小的放后面去
				memcpy(&temp_node, current_node, sizeof(yolo_result));
				
				memcpy(current_node, comparable_node, sizeof(yolo_result));
				
				memcpy(comparable_node, &temp_node, sizeof(yolo_result));
				
				current_node->next = current_next_node; // 链表接好
				
				comparable_node->next = comparable_next_node;

				comparable_node = comparable_node->next; //更新位置，因为当前节点已经小于current_node ，不必再做比较
			}
			
		}
		
		current_node = current_node->next;
	}
}


void yolo_nms(yolo_result *output_result, float iou_threshold){ 

	yolo_result *comparable_node = NULL; // 右节点，挨个指向右边所有节点
	
	yolo_result *comparable_former_node = NULL;
	
	yolo_result *current_node = output_result; // 左节点，其与右边每个节点做比较

	yolo_result *temp_node = NULL;

	float overlap_left_x = 0.0f;

	float overlap_left_y = 0.0f;

	float overlap_right_x = 0.0f;

	float overlap_right_y = 0.0f;

	float current_area = 0.0f, comparable_area = 0.0f, overlap_area = 0.0f;

	float nms_ratio = 0.0f;

	float overlap_w = 0.0f, overlap_h = 0.0f;
	
	// yolo v5 的 nms 实现很优雅，我没在这里用，我在博客里介绍一下
	while (current_node != NULL){
	
		comparable_node = current_node->next;

		comparable_former_node = current_node;
		//printf("current_node->score = %f\n", current_node->score);
		current_area = (current_node->right_down_x - current_node->left_up_x) * (current_node->right_down_y - current_node->left_up_y);
	
		while (comparable_node != NULL){
			
			if (current_node->class_index != comparable_node->class_index){ // 如果类别不一致，没必要做 nms
			
				comparable_former_node = comparable_node;

				comparable_node = comparable_node->next;
				continue;
			}
			//printf("comparable_node->score = %f\n", comparable_node->score);
			comparable_area = (comparable_node->right_down_x - comparable_node->left_up_x) * (comparable_node->right_down_y - comparable_node->left_up_y);

			overlap_left_x = YOLO_MAX(current_node->left_up_x, comparable_node->left_up_x);
			overlap_left_y = YOLO_MAX(current_node->left_up_y, comparable_node->left_up_y);

			overlap_right_x = YOLO_MIN(current_node->right_down_x, comparable_node->right_down_x);
			overlap_right_y = YOLO_MIN(current_node->right_down_y, comparable_node->right_down_y);

			overlap_w = YOLO_MAX((overlap_right_x - overlap_left_x), 0.0F);
			overlap_h = YOLO_MAX((overlap_right_y - overlap_left_y), 0.0F);
			overlap_area = YOLO_MAX((overlap_w * overlap_h), 0.0f); // 重叠区域面积

			nms_ratio = overlap_area / (current_area + comparable_area - overlap_area);
			
			if (nms_ratio > iou_threshold){ // 重叠过大，去掉
			
				temp_node = comparable_node;
				
				comparable_node = comparable_node->next;

				comparable_former_node->next = comparable_node; // 链表接好
				
				free(temp_node);
			}else{
				
				comparable_former_node = comparable_node;
				
				comparable_node = comparable_node->next;
			}
			
		}
		//printf("loop end \n");
		current_node = current_node->next;
	}
}

void printf_result(yolo_result *temp){
	printf("--------------------\n");

	while (temp != NULL){
		
		printf("output_result->left_up_x = %f\t", temp->left_up_x);
		printf("output_result->left_up_y = %f\n", temp->left_up_y);

		printf("output_result->right_down_x = %f\t", temp->right_down_x);
		printf("output_result->right_down_y = %f\n", temp->right_down_y);

		printf("output_result->class_index = %d\t", temp->class_index);
		printf("output_result->score = %f\n\n", temp->score);
		
		temp = temp->next;
	}
	printf("--------------------\n");
}

void release_result(yolo_result *output_result){

	yolo_result *temp = NULL;

	while (output_result != NULL){
		
		temp = output_result;
		
		output_result = output_result->next;
	
		free(temp);
	}
}

// 图片推理版本
int yolo_image_inference(char *pcModelName, float confidence_threshold, float iou_threshold, char *pcSrcFile){

    HI_S32 s32Ret = HI_SUCCESS;  // HI_SUCCESS = 0
    /*Set configuration parameter*/
    HI_U32 u32PicNum = 1;
    /*Set configuration parameter*/
    stNnieCfg.u32MaxInputNum = u32PicNum; //max input image num in each batch ; stNnieCfg = {0}
    stNnieCfg.u32MaxRoiNum = 0;
    stNnieCfg.aenNnieCoreId[0] = SVP_NNIE_ID_0;//set NNIE core

	yolo_result *output_result = NULL;

    /*Sys init*/
    SAMPLE_COMM_SVP_CheckSysInit();

    /*CNN Load model*/
    SAMPLE_SVP_TRACE_INFO("Cnn Load model!\n");
    s32Ret = SAMPLE_COMM_SVP_NNIE_LoadModel(pcModelName,&s_stDetModel); // sample_comm_nnie.c 文件下定义 ；全局变量 ：结构体 s_stDetModel 初始全为0，里面储存了模型网络结构的指针 。
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret,CNN_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_COMM_SVP_NNIE_LoadModel failed!\n");
    
    SAMPLE_SVP_TRACE_INFO("Cnn parameter initialization!\n");
    s_stDetNnieParam.pstModel = &s_stDetModel.stModel; // 网络模型结构体 stModel 。
    s32Ret = SAMPLE_SVP_NNIE_Cnn_ParamInit(&stNnieCfg, &s_stDetNnieParam); // 初始化 NNIE 硬件层参数
    
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret,CNN_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_Cnn_ParamInit failed!\n");
	
    SAMPLE_SVP_TRACE_INFO("NNIE AddTskBuf!\n");
	
    /*record tskBuf*/
    s32Ret = HI_MPI_SVP_NNIE_AddTskBuf(&(s_stDetNnieParam.astForwardCtrl[0].stTskBuf)); // 记录 TskBuf 地址信息 ,用于减少内核态内存映射次数 ,提升效率 。
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret,CNN_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,HI_MPI_SVP_NNIE_AddTskBuf failed!\n");
	
    SAMPLE_SVP_TRACE_INFO("NNIE AddTskBuf end!\n");

	stNnieCfg.pszPic= pcSrcFile;
    s32Ret = HI_SUCCESS;
    SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S stInputDataIdx = {0};
    SAMPLE_SVP_NNIE_PROCESS_SEG_INDEX_S stProcSegIdx = {0};

    /*Fill src data*/
    stInputDataIdx.u32SegIdx = 0;
    stInputDataIdx.u32NodeIdx = 0;
    s32Ret = SAMPLE_SVP_NNIE_FillSrcData(&stNnieCfg,&s_stDetNnieParam,&stInputDataIdx); // 加载图片
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret,CNN_FAIL_1,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_FillSrcData failed!\n");
	
    /*NNIE process(process the 0-th segment)*/
    stProcSegIdx.u32SegIdx = 0;
    s32Ret = SAMPLE_SVP_NNIE_Forward(&s_stDetNnieParam, &stInputDataIdx, &stProcSegIdx, HI_TRUE); // 数据前向推理，一帧开始
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret,CNN_FAIL_1,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_Forward failed!\n");
    SAMPLE_SVP_TRACE_INFO("Forward!\n");
    
    /*Software process*/ // 处理输出结果，对结果进行筛选，下面才是不一样的地方，上面是固定模板 。
    
	unsigned int resltu_num = yolo_result_process(&s_stDetNnieParam, strides, anchor_grids, map_size, &output_result, confidence_threshold); // 后处理
	
	//printf_result(output_result);

	if(output_result != NULL){
		yolo_result_sort(output_result); // 输出结果排序，方便下面的 nms 处理

        //printf_result(output_result);
	
	    yolo_nms(output_result, iou_threshold); // nms
	}
	
	printf_result(output_result); // 打印结果看下，这里打印的已经是可以输出画框的结果了
	
	release_result(output_result); // 释放内存，一帧结束
	
    SAMPLE_SVP_NNIE_Cnn_Deinit(&s_stDetNnieParam, &s_stDetModel);
    memset(&s_stDetNnieParam,0,sizeof(SAMPLE_SVP_NNIE_PARAM_S));
    memset(&s_stDetModel,0,sizeof(SAMPLE_SVP_NNIE_MODEL_S));
    SAMPLE_COMM_SVP_CheckSysExit();
	
	return 0;
CNN_FAIL_1:
	/*Remove TskBuf*/
	SAMPLE_SVP_TRACE_INFO("Why1 \n");
	s32Ret = HI_MPI_SVP_NNIE_RemoveTskBuf(&(s_stDetNnieParam.astForwardCtrl[0].stTskBuf));
	SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret,CNN_FAIL_0,SAMPLE_SVP_ERR_LEVEL_ERROR,
		"Error,HI_MPI_SVP_NNIE_RemoveTskBuf failed!\n");


CNN_FAIL_0:
	SAMPLE_SVP_TRACE_INFO("Why \n");
	SAMPLE_SVP_NNIE_Cnn_Deinit(&s_stDetNnieParam, &s_stDetModel);
	SAMPLE_COMM_SVP_CheckSysExit();
	return 0;

}
