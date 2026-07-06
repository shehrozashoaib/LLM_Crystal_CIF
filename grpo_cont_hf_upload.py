from huggingface_hub import HfApi
import os
api=HfApi(); REPO="shehrozashoaib/LLM_Crystal_CIF"; base="experiments/grpo_r32_from3000_continuous"
existing=set(api.list_repo_files(REPO))
for label,parent in [("final",f"{base}/final_model"),("best",f"{base}/best_model")]:
    key=f"grpo_r32_from3000_continuous/{label}/model/adapter_model.safetensors"
    if key in existing: print(f"[skip] cont {label}"); continue
    print(f"[upload] cont {label} ...")
    api.upload_folder(repo_id=REPO, folder_path=f"{parent}/model", path_in_repo=f"grpo_r32_from3000_continuous/{label}/model", commit_message=f"Add GRPO-continuous {label} adapter")
    api.upload_folder(repo_id=REPO, folder_path=f"{parent}/tokenizer", path_in_repo=f"grpo_r32_from3000_continuous/{label}/tokenizer", commit_message=f"Add GRPO-continuous {label} tokenizer")
bi=f"{base}/best_model/best_info.json"
if os.path.isfile(bi): api.upload_file(repo_id=REPO, path_or_fileobj=bi, path_in_repo="grpo_r32_from3000_continuous/best_info.json", commit_message="Add GRPO-continuous best_info")
print("CONT HF DONE")
