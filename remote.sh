conda activate ml8

jupyter notebook \
	--NotebookApp.iopub_data_rate_limit=1.0e10 \
	--no-browser \
	--ip 192.168.1.131 \
	--port 8880

