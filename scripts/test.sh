#model=siamrpn_mobilev2_l234_dwxcorr
#model=siamrpn_r50_l234_dwxcorr_otb
model=siamrpn_r50_l234_dwxcorr_lt
#model=siamrpn_alex_dwxcorr
#vid=w1.mp4
#vid=bag.avi
#vid=2.mp4
#vid=2018-10-01-09-00-22-420620.mp4
#vid=2018-10-01-09-03-35-421138.mp4
#vid=2018-10-01-09-02-17-420839.mp4
#vid=2018-10-01-09-00-39-420675.mp4
vid=2018-10-01-09-00-22-420620.mp4
#vid=2018-10-01-18-04-08-440564.mp4
#vid=2018-10-01-18-01-12-440249.mp4
#vid=2018-10-01-09-05-30-421443.mp4
#vid=2018-10-01-09-03-28-421116.mp4
#vid=2018-10-01-09-03-35-421138.mp4 #skip
#vid=2018-10-01-09-02-12-420824.mp4
#python ../tools/demo_mt.py --config ../experiments/$model/config.yaml --snapshot ../experiments/$model/model.pth --video ../demo/$vid
#python ../tools/demo.py --config ../experiments/$model/config.yaml --snapshot ../experiments/$model/model.pth --video ../demo/$vid
python ../tools/track_annot.py --config ../experiments/$model/config.yaml --snapshot ../experiments/$model/model.pth --video ../demo/$vid
#
#
#model=siamrpn_alex_dwxcorr_4gpu
#python ../tools/demo.py --config ../experiments/$model/config_deploy.yaml --snapshot ../experiments/$model/snapshot/checkpoint_e27.pth --video ../demo/$vid

