import stage2.stage2_image_processing
import stage2.stage2_excel_merge
import stage2.stage2_kfold , stage2.stage2_kfold_two, stage2.stage2_kfold_three
import protien

target_lsit = ['PD','SP','GA']      # 'PD','SP','GA'
stage2.stage2_image_processing.main(target_lsit)
stage2.stage2_excel_merge.main(target_lsit)
#protien.main()
#stage2.stage2_kfold.main(target_lsit)
#stage2.stage2_kfold_two.main(target_lsit)
stage2.stage2_kfold_three.main(target_lsit)