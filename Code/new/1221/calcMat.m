% 参数为相机标定导出的变量
% 打印出的内容可直接复制至.py文件
function [proj_mats_left, proj_mats_right] = calcMat(stereoParam)
    fprintf("\n");
    proj_mats_left = transpose(stereoParam.CameraParameters1.IntrinsicMatrix);
    proj_mats_left(1:3, 4) = 0;
    T = stereoParam.RotationOfCamera2;
    T(4,4) = 1;
    T(1:3, 4) = transpose(stereoParam.TranslationOfCamera2);
    proj_mats_right = proj_mats_left * inv(T);
%     proj_mats_right = proj_mats_left * T;
    fprintf("proj_mats_left = np.array([\n");
    for i = [1:3]
        fprintf("[");
        for j = [1:4]
            fprintf("%f", proj_mats_left(i, j));
            if (j < 4)
                fprintf(", ");
            end
        end
        if (i < 3) 
            fprintf("],\n");
        else
            fprintf("]], dtype=float)\n");
        end
    end
    fprintf("\n");
    
    fprintf("proj_mats_right = np.array([\n");
    for i = [1:3]
        fprintf("[");
        for j = [1:4]
            fprintf("%f", proj_mats_right(i, j));
            if (j < 4)
                fprintf(", ");
            end
        end
        if (i < 3) 
            fprintf("],\n");
        else
            fprintf("]], dtype=float)\n");
        end
    end
    fprintf("\n");
end