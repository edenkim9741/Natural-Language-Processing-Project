import torch
from transformers import BertModel, BertTokenizer
import numpy as np
import argparse

class BertFeatureExtractor:
    def __init__(self, model_name="bert-base-multilingual-cased"):
        """
        BERT 모델을 이용한 Feature Extractor 초기화
        
        Args:
            model_name (str): 사용할 BERT 모델 이름 
                              (기본값: bert-base-multilingual-cased - 다국어 지원)
        """
        # GPU 사용 가능 여부 확인
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # 토크나이저와 모델 로드
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()  # 평가 모드로 설정
    
    def extract_features(self, sentences, pooling_strategy="cls"):
        """
        문장 목록에서 BERT 임베딩 추출
        
        Args:
            sentences (list): 임베딩을 추출할 문장 리스트
            pooling_strategy (str): 임베딩 풀링 전략 ('cls', 'mean', 'max')
            
        Returns:
            numpy.ndarray: 문장 임베딩 배열
        """
        # 입력 인코딩
        encoded_input = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # 인코딩된 입력을 디바이스로 이동
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        
        # 그래디언트 계산 없이 추론 실행
        with torch.no_grad():
            outputs = self.model(**encoded_input)
        
        # 마지막 히든 스테이트
        last_hidden_state = outputs.last_hidden_state
        
        # 풀링 전략에 따라 임베딩 추출
        if pooling_strategy == "cls":
            # [CLS] 토큰 임베딩 사용 (첫 번째 토큰)
            embeddings = last_hidden_state[:, 0]
        elif pooling_strategy == "mean":
            # 모든 토큰의 평균 임베딩
            # attention_mask로 패딩 토큰 제외
            attention_mask = encoded_input['attention_mask']
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = torch.sum(input_mask_expanded, 1)
            embeddings = sum_embeddings / sum_mask
        elif pooling_strategy == "max":
            # 모든 토큰의 최대값 임베딩
            attention_mask = encoded_input['attention_mask']
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            embeddings = torch.max(last_hidden_state * input_mask_expanded, 1)[0]
        else:
            raise ValueError("pooling_strategy는 'cls', 'mean', 'max' 중 하나여야 합니다.")
            
        # CPU로 이동 후 numpy 배열로 변환
        return embeddings.cpu().numpy()


# 사용 예시
if __name__ == "__main__":
    # 예시 문장들
    sentences = [
        "안녕하세요, 반갑습니다.",
        "자연어 처리는 인공지능의 중요한 분야입니다.",
        "BERT는 트랜스포머 기반 언어 모델입니다."
    ]
    
    # Feature Extractor 초기화
    extractor = BertFeatureExtractor()
    
    # CLS 토큰 임베딩 추출
    cls_embeddings = extractor.extract_features(sentences, pooling_strategy="cls")
    print(f"CLS 임베딩 shape: {cls_embeddings.shape}")
    
    # 평균 풀링 임베딩 추출
    mean_embeddings = extractor.extract_features(sentences, pooling_strategy="mean")
    print(f"평균 풀링 임베딩 shape: {mean_embeddings.shape}")
    
    # 특정 문장의 임베딩 값 확인 (처음 5개 차원만)
    print(f"첫 번째 문장의 임베딩 (처음 5개 차원): {cls_embeddings[0][:5]}")
    
    # 두 문장 간 코사인 유사도 계산
    from sklearn.metrics.pairwise import cosine_similarity
    
    similarity = cosine_similarity([cls_embeddings[0]], [cls_embeddings[1]])
    print(f"첫 번째와 두 번째 문장의 코사인 유사도: {similarity[0][0]}")