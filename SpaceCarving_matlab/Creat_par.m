clear all

data_path = '../data/SyntheticCase1/';
model_path = [data_path, '/colmap'];
output_fname = [data_path, '/masks/frame_par.txt'];
selected_image_dir = [data_path, '/masks/'];

scaling_factor = 0.09;

%% Read in model
[cameras, images, points3D] = read_model(model_path);
%% Camera intrinsics
camera = cameras(1);
focalLength    = [camera.params(1), camera.params(1)];
principalPoint = [camera.params(2), camera.params(3)];
imageSize      = [camera.height, camera.width];
intrinsics = cameraIntrinsics(focalLength,principalPoint,imageSize, 'RadialDistortion', [camera.params(4), camera.params(4)]);
%intrinsics = cameraIntrinsics(focalLength,principalPoint,imageSize);
K = intrinsics.IntrinsicMatrix';

camera = cameras(1);
k11 = K(1, 1);
k12 = K(1, 2);
k13 = K(1, 3);

k21 = K(2, 1);
k22 = K(2, 2);
k23 = K(2, 3);

k31 = K(3, 1);
k32 = K(3, 2);
k33 = K(3, 3);


%% Generate depth images
keys = images.keys;
L = images.length;

fileID = fopen(output_fname,'w');

num_frames = 0;
for k = 1:L    
    %for k = 1
    image_id = keys{k};
    image = images(image_id);    
    file_name = strrep([selected_image_dir, 'mask_', image.name], ".bmp", ".png");
    
    if ( exist(file_name)==2 )
        num_frames = num_frames + 1;
    end
end

fprintf(fileID, "%d\n", num_frames);
for k = 1:L
    %for k = 1
    image_id = keys{k};
    image = images(image_id);
    
    %file_name = [selected_image_dir, image.name, '.png'];
    file_name = strrep([selected_image_dir, 'mask_', image.name], ".bmp", ".png.png");
    if ( exist(file_name)==2 )
        % R = image.R';
        % t = -R*image.t*0.08;
        R=image.R;
        %t=image.t;
        t=image.t*scaling_factor;
        r11 = R(1, 1);
        r12 = R(1, 2);
        r13 = R(1, 3);

        r21 = R(2, 1);
        r22 = R(2, 2);
        r23 = R(2, 3);

        r31 = R(3, 1);
        r32 = R(3, 2);
        r33 = R(3, 3);

        t1 = t(1);
        t2 = t(2);
        t3 = t(3);

        %fprintf(fileID,'%s %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f \n',['frame_',image.name], k11, k12, k13, k21, k22, k23, k31, k32, k33, r11, r12, r13, r21, r22, r23, r31, r32, r33, t1, t2, t3);
        fprintf(fileID,'%s %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f \n',file_name, k11, k12, k13, k21, k22, k23, k31, k32, k33, r11, r12, r13, r21, r22, r23, r31, r32, r33, t1, t2, t3);
    
    end
end
fclose(fileID);
