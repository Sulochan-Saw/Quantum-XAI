# Manual process to add Member 7 files to the existing Quantum-XAI repository

Repository: https://github.com/Sulochan-Saw/Quantum-XAI

## Before uploading
1. Open the repository link and sign in to GitHub.
2. Check the existing folder structure and README so you do not overwrite another member's work.
3. Download and unzip `Quantum_XAI_Member7_Package.zip` on your computer.
4. In the extracted folder, review `README.md` and change the contributor display name only if your team uses a different exact spelling.
5. Keep the synthetic demo CSV clearly labelled as synthetic. Do not replace the team's real dataset or upload private patient data.

## Upload using the GitHub website (no Git commands)
1. Open https://github.com/Sulochan-Saw/Quantum-XAI
2. Select the branch dropdown (usually `main`). Create a new branch named `member7-app-docs` if you have permission. If your team requires pull requests, use this branch.
3. Click **Add file** → **Upload files**.
4. Open the extracted package folder on your computer. Drag the `app`, `docs`, and `data` folders plus `README.md` and `requirements.txt` into the upload area. Also upload `MANUAL_GITHUB_UPLOAD.md`.
5. If GitHub does not preserve empty folders, that is normal; these folders contain files.
6. Scroll to **Commit changes**. Use a message such as `Add Member 7 application and documentation contribution`.
7. If uploading to a new branch, choose **Create a new branch for this commit and start a pull request** (if shown), then submit the pull request for the project lead to review.
8. If your team has asked you to commit directly to `main`, select that only after confirming with the repository owner.

## If a file already exists
- Do not blindly replace it.
- Open the existing file and compare it with the package version.
- For an existing README, copy only the relevant Member 7 section into the current README, preserving all existing members and links.
- If `requirements.txt` exists, merge only missing dependencies (`streamlit`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `pennylane`) rather than replacing the file.
- If `app/` or `docs/` already exists, upload only new files or agree a unique subfolder name with the team.

## After uploading
1. Open the new commit or pull request and check that all files appear.
2. Ask the project lead to review and merge the pull request.
3. In the project README, list your contribution under the existing team roster; do not remove or rename other contributors.
4. Share the commit/PR link with your teammates.

## Troubleshooting
- **404:** Confirm the repository URL is correct and that your GitHub account has access. A private repository requires membership or an authorized GitHub App.
- **No Add file button:** You may have read-only access; ask the repository owner to grant write access.
- **Upload size limit:** GitHub blocks individual files over 100 MB. Keep large datasets out of the repository and use the team's approved storage.
