PS C:\Users\Lenovo\Desktop\ml> python vertex_ai_training_main.py
Creating TabularDataset
Create TabularDataset backing LRO: projects/1025366590832/locations/asia-south1/datasets/7837081388275728384/operations/2009713191772225536
TabularDataset created. Resource name: projects/1025366590832/locations/asia-south1/datasets/7837081388275728384
To use this TabularDataset in another session:
ds = aiplatform.TabularDataset('projects/1025366590832/locations/asia-south1/datasets/7837081388275728384')
Training script copied to:
gs://mla1/aiplatform-2025-09-28-00:55:38.038-aiplatform_custom_trainer_script-0.1.tar.gz.
Training Output directory:
gs://mla1/aiplatform-custom-training-2025-09-28-00:55:42.999
No dataset split provided. The service will use a default split.
View Training:
https://console.cloud.google.com/ai/platform/locations/asia-south1/training/6830481241825869824?project=1025366590832
CustomTrainingJob projects/1025366590832/locations/asia-south1/trainingPipelines/6830481241825869824 current state:
3
CustomTrainingJob projects/1025366590832/locations/asia-south1/trainingPipelines/6830481241825869824 current state:
3
CustomTrainingJob projects/1025366590832/locations/asia-south1/trainingPipelines/6830481241825869824 current state:
3
CustomTrainingJob projects/1025366590832/locations/asia-south1/trainingPipelines/6830481241825869824 current state:
3
CustomTrainingJob projects/1025366590832/locations/asia-south1/trainingPipelines/6830481241825869824 current state:
3
Traceback (most recent call last):
  File "C:\Users\Lenovo\Desktop\ml\vertex_ai_training_main.py", line 37, in <module>
    main()
  File "C:\Users\Lenovo\Desktop\ml\vertex_ai_training_main.py", line 34, in main
    trainer.train_model(X_train, y_train)
  File "C:\Users\Lenovo\Desktop\ml\src\models\vertex_ai_trainer.py", line 101, in train_model
    model = job.run(
            ^^^^^^^^
  File "C:\Users\Lenovo\AppData\Roaming\Python\Python312\site-packages\google\cloud\aiplatform\training_jobs.py", line 3480, in run
    return self._run(
           ^^^^^^^^^^
  File "C:\Users\Lenovo\AppData\Roaming\Python\Python312\site-packages\google\cloud\aiplatform\base.py", line 862, in wrapper
    return method(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Lenovo\AppData\Roaming\Python\Python312\site-packages\google\cloud\aiplatform\training_jobs.py", line 4298, in _run
    model = self._run_job(
            ^^^^^^^^^^^^^^
  File "C:\Users\Lenovo\AppData\Roaming\Python\Python312\site-packages\google\cloud\aiplatform\training_jobs.py", line 855, in _run_job
    model = self._get_model(block=block)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Lenovo\AppData\Roaming\Python\Python312\site-packages\google\cloud\aiplatform\training_jobs.py", line 942, in _get_model
    self._block_until_complete()
  File "C:\Users\Lenovo\AppData\Roaming\Python\Python312\site-packages\google\cloud\aiplatform\training_jobs.py", line 985, in _block_until_complete
    self._raise_failure()
  File "C:\Users\Lenovo\AppData\Roaming\Python\Python312\site-packages\google\cloud\aiplatform\training_jobs.py", line 1002, in _raise_failure
    raise RuntimeError("Training failed with:\n%s" % self._gca_resource.error)
RuntimeError: Training failed with:
code: 8
message: "The following quota metrics exceed quota limits: aiplatform.googleapis.com/custom_model_training_cpus"