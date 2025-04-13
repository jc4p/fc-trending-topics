rsync -avz --progress --exclude="*.parquet" --exclude="**/.ipynb_checkpoints/" --exclude="**/.git/" ubuntu@64.181.249.47:~/fc-trending-topics/ ./
