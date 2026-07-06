from huggingface_hub import HfApi
api=HfApi(); REPO="shehrozashoaib/LLM_Crystal_CIF"; base="experiments/grpo_r32_from3000_discrete"
existing=set(api.list_repo_files(REPO))
for label,parent in [("final",f"{base}/final_model"),("best",f"{base}/best_model")]:
    if f"grpo_r32_from3000_discrete/{label}/model/adapter_model.safetensors" in existing:
        print(f"[skip] grpo {label} already on repo"); continue
    print(f"[upload] grpo {label} ...")
    api.upload_folder(repo_id=REPO, folder_path=f"{parent}/model", path_in_repo=f"grpo_r32_from3000_discrete/{label}/model", commit_message=f"Add GRPO {label} adapter")
    api.upload_folder(repo_id=REPO, folder_path=f"{parent}/tokenizer", path_in_repo=f"grpo_r32_from3000_discrete/{label}/tokenizer", commit_message=f"Add GRPO {label} tokenizer")
import os
bi=f"{base}/best_model/best_info.json"
if os.path.isfile(bi): api.upload_file(repo_id=REPO, path_or_fileobj=bi, path_in_repo="grpo_r32_from3000_discrete/best_info.json", commit_message="Add GRPO best_info")
print("GRPO upload done")
