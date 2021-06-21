This project has the goal to compare the result of different tools in the task of OCR post processing (error correction)

It is organized following a set of steps described bellow:
	- Create a dataset with the script of error\_insertion (github ref)
		- The dataset consists of abstracts text, and their error inserted versions
		The original number of abstract is more than 20k texts, so, in order to reduce the time of processing, we select randomly near of 10% of the original data (2k abstracts-dataset-abstracts)
	- Use the following tools to correct dataset-abstracts
		- Socrates
		- Ochre
		- Sysmpell
		- Aspell
	- Use dinglehopper to measure the performance of each tool .- 
	()
		example command:
			dinglehopper folder\_gt folder\_ocr --progress
		Observation .- Dinglehopper has high memory usage requirement for few paragraph of texts 
	- Use ocrEvaluation as option to replace dinglehopper
	(https://github.com/impactcentre/ocrevalUAtion/)
		- java -cp ocrevalUAtion-1.3.4.jar eu.digitisation.Main -gt ./gt_teseractpuc/ -ocr ./ochre_seq25_teseractpuc/ -o data_corrig_teseract/ochre_seq25.xml -ip -ic


Warning
	- For this experiment, it was used ngrams of folha dataset (it should be replaced with a genericportuguese ngrams)
	- The vocab used to create the generic trie, it was the vocab\_union (This should not need to be replaced because it is already generic)
	- the function removeall it used from the utils repository (https://github.com/dannysv/utils)
	clone and change the folder name to mutils to skip the reference error with the utils script that alredy exist in the current project

	- The script corrigir\_socrates\_filtro.py should be placed on the correct folder (classificador\_folha\_completo), this folder contains the dependencies as well as the bert models and other necessary tools

	- The file requirements\_socrates.txt contains the list of dependencies for running the socrates script.

