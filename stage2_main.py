import stage2_image_processing
import stage2_excel_merge
import stage2_kfold , stage2_kfold_two, stage2_kfold_three

target_lsit = ['PD','SP','GA']      # 'PD','SP','GA'
#stage2_image_processing.main(target_lsit)
#stage2_excel_merge.main(target_lsit)
#stage2_kfold.main(target_lsit)
#stage2_kfold_two.main(target_lsit)
stage2_kfold_three.main(target_lsit)