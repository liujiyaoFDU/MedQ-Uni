
# Get the current Linux machine's ID (hostname) and the account name (whoami)
HOSTNAME=$(hostname)
USER=$(whoami)

# Get the list of modified files from git status and store it in a variable
CHANGES=$(git status --short | grep -E '^[ MARD]' | awk '{print $2}')

# If no changes are detected, exit the script
if [ -z "$CHANGES" ]; then
  echo "No changes to commit."
  exit 0
fi

# Generate the commit title (short summary) and detailed message (list of modified files)
COMMIT_TITLE="Update on $(date +"%Y-%m-%d %H:%M:%S") by $USER@$HOSTNAME"
COMMIT_DETAILS="Modified files:\n$CHANGES"

# Create a temporary file to store the commit message
COMMIT_FILE=$(mktemp)

# Write the commit title and detailed message to the temporary file
echo -e "$COMMIT_TITLE\n\n$COMMIT_DETAILS" > "$COMMIT_FILE"

# Display the generated commit message
echo "Committing with message:"
cat "$COMMIT_FILE"

# Stage all modified files for commit
git add -A .

# Commit the changes using the message from the temporary file
git commit -F "$COMMIT_FILE"

# Push the changes to the remote repository
git push

# Show git status after pushing changes
git status

# Remove the temporary file after the commit
rm "$COMMIT_FILE"