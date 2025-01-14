clc
clear all
close all

% wczytywanie zdj��
%%-----------------------
face_basis = [];
for i = 1:400
    clc
    i
    name = ['foto (', int2str(i), ').jpg'];
    A = imread(name);
    A = A(:,:,1);
    A = reshape(A, 193  * 162,1);
    face_basis = [face_basis A];
end

% wy�wietlanie wektor�w szczeg�lnych
%%
which_singular_vect = 12;
face_basis_double = double(face_basis);
[S V D] = svd(face_basis_double,'econ');
disp(size(S))
clc
s1 = abs(reshape(S(:,which_singular_vect), 193,162));
s1 = s1/max(max(s1));
s1 = s1 * 255;
s1 = uint8(abs(s1));
imshow(s1);

% zdj�cia przed i po kompresji. 
%%
N = 60;  % ile wektor�w szczegonlych? 
which_photo = 204;
asd = S(:,1:N) * (face_basis_double(:,which_photo)' * S(:,1:N))';   

s1 = abs(reshape(asd, 193,162));
s1 = s1/max(max(s1));
s1 = s1 * 255;
s1 = uint8(abs(s1));
figure
subplot(1,2,2)
imshow(s1);

subplot(1,2,1)
imshow( reshape(face_basis(:,which_photo), 193,162)) ;













