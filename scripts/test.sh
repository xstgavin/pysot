#model=siamrpn_mobilev2_l234_dwxcorr
#model=siamrpn_r50_l234_dwxcorr_otb
#model=siamrpn_r50_l234_dwxcorr_lt
#model=siamrpn_alex_dwxcorr
#vid=w1.mp4
#vid=bag.avi
vid=2.mp4
#python ../tools/demo_mt.py --config ../experiments/$model/config.yaml --snapshot ../experiments/$model/model.pth --video ../demo/$vid
model=siamrpn_alex_dwxcorr_16gpu
python ../tools/demo.py --config ../experiments/$model/config_deploy.yaml --snapshot ../experiments/$model/snapshot/checkpoint_e14.pth --video ../demo/$vid

