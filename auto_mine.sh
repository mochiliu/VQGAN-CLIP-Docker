#!/bin/bash

script_order=$(ScriptToOrdering.sh $script_name)
if [ "$step_to_start" -lt "$script_order" ]; then
	#submit job to cluster
	PROCESS_ID=$(sbatch -N1 -n1 --mem-per-cpu=16000M -t23:59:00 RunMatlabJVM.sh $folder_name $script_name) #  
	UpdateLog.sh $folder_name $script_name ${PROCESS_ID##* } SUBMIT Awaiting_Resources #update the log
	while squeue -u $user_name | grep -q -w ${PROCESS_ID##* }; do sleep 10; done #wait until job finishes
	#check if the operation completed succesfully
	exit_command=$(CompletionCheck.sh $folder_name $script_name ${PROCESS_ID##* }) 
	if [ "$exit_command" == "EXIT" ]; then
		echo $script_name' failed, exiting in '$folder_name
		exit
	fi
fi
sh -c 'echo $$; exec myCommand'
