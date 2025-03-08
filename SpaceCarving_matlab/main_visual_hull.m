% dataset path
% data_path = '../data/SyntheticCase1/';

%%
%% run Creat_par.m first
%%
data_dir = [data_path, '/masks/']; 

% paremeters 
silhouette_thresold = 100/255.0;
voxel_grid_size_xyz = [100, 100, 100];

%load data
DL = DataLoader(data_dir);
DL = DL.LoadCameraParams();
DL = DL.LoadImages();
DL = DL.CalcFOVUnion();

%calc visual hull
VH = VisualHull(DL);
VH = VH.ExtractSilhoueteFromImages(silhouette_thresold);
VH = VH.CreateVoxelGrid(voxel_grid_size_xyz);
VH = VH.ProjectVoxelsToSilhouette();

% show results
figure; 
VH.ShowVH3D();
% hold on;
% DL.PlotFOV([1 2 3]);

figure; 
VH.ShowVH3D();
hold on;
%DL.PlotFOV([1,6,20], 'blue', 0.1);
DL.PlotFOV([], 'blue', 0.01);
DL.PlotBoudningVolume('green', 1);

%figure; VH.ShowVH2DGrid(180);

%save results to stl file
VH.SaveGeoemtry2STL()

%% postprocessing
pts = VH.voxels_voted(VH.voxels_voted(:, 4) > 20, 1:3);
newpts = pts/scaling_factor;
[zmin, idmin] = min(newpts(:, 3));
[zmax, idmax] = max(newpts(:, 3));
z = (zmin + zmax)/2;
[~, id] = min( abs(newpts(:, 3) - z));

[newpts(idmax, :), newpts(id, 1:2), newpts(idmin, :)] 