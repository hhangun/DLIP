clc; clear all; close all;
data = load("counting_result.txt");
data = data(1:1501,:);
true = load('LAB_Parking_counting_result_answer_student_modified.txt');

compare_data  = data(:,2)-true(:,2);

accuracy1 = mean(13-abs(compare_data))/13*100;
falsenegative = find(compare_data>0);
truepositive = find(compare_data==0);
falsepositive= find(compare_data<0);

accuracy = length(truepositive)/1501
precision= length(truepositive)/(length(truepositive)+length(falsepositive))
recall= length(truepositive)/(length(truepositive)+length(falsenegative))
f1    = 2*precision*recall/(precision+recall)