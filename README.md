# neural topic  model

### This is a pure neural topic model basd on this architecture:
https://github.com/yuewang-cuhk/TAKG
### If you want to run the codes, just run the following commands:
python train_main.py -data_tag StackExchange_s150_t10 -only_train_ntm -ntm_warm_up_epochs ${number_epochs}
**note:** you have to specify the training epochs
