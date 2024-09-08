# UAV Safe Landing System Based on Deep Reinforcement Learning
本論文目標為以深度圖及灰階圖當作輸入，透過CM-VAE模型對輸入進行特徵學習、降燥及降維後再做為深度強化學習的輸入，深度強化學習模型輸出適當離散動作，使無人機能盡速降落在平坦且安全的區域，並且能在降落期間避開障礙物。模型訓練部分，在基於AirSim的模擬場景中進行多次的學習，並在不同於訓練環境的場景下進行模型的評估與比較。

## System Architecture

![系統架構圖](https://github.com/user-attachments/assets/98dee07c-3dce-4fee-88db-7c8d17d5ab7c)

## Training Flowchart

![訓練流程](https://github.com/user-attachments/assets/0198d853-772a-4e6c-a57a-9a30c3561257)

## Cross Modality Variational AutoEncoder Architecture
- Encdoer
![CMVAE模型架構](https://github.com/user-attachments/assets/1cdfc4c7-5226-4c26-890f-c75cf096016e)
- Decoder
![CMVAE_decoder](https://github.com/user-attachments/assets/38836620-e7b3-402b-9f71-f25ae129c077)

## Training Result

## Test Result Demo 
https://youtu.be/xtk49do-xW0
## Realfly Test
https://www.youtube.com/watch?v=oULu6L_CSug
