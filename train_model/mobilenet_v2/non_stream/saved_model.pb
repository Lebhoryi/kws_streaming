฿7
ัฃ
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
พ
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02v2.3.0-rc2-23-gb36436b0878าต,
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:( *
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:( *
dtype0

batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namebatch_normalization/beta

,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
: *
dtype0

batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!batch_normalization/moving_mean

3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
: *
dtype0

#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#batch_normalization/moving_variance

7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
: *
dtype0

conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
: 0*
dtype0

batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*+
shared_namebatch_normalization_1/beta

.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
:0*
dtype0

!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*2
shared_name#!batch_normalization_1/moving_mean

5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:0*
dtype0
ข
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*6
shared_name'%batch_normalization_1/moving_variance

9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
:0*
dtype0
ฆ
!depthwise_conv2d/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*2
shared_name#!depthwise_conv2d/depthwise_kernel

5depthwise_conv2d/depthwise_kernel/Read/ReadVariableOpReadVariableOp!depthwise_conv2d/depthwise_kernel*&
_output_shapes
:0*
dtype0

batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*+
shared_namebatch_normalization_2/beta

.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
:0*
dtype0

!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*2
shared_name#!batch_normalization_2/moving_mean

5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
:0*
dtype0
ข
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*6
shared_name'%batch_normalization_2/moving_variance

9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
:0*
dtype0

conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0 * 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
:0 *
dtype0

batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_3/gamma

/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes
: *
dtype0

batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_3/beta

.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes
: *
dtype0

!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_3/moving_mean

5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes
: *
dtype0
ข
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_3/moving_variance

9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes
: *
dtype0

conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0* 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
: 0*
dtype0

batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*+
shared_namebatch_normalization_4/beta

.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
_output_shapes
:0*
dtype0

!batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*2
shared_name#!batch_normalization_4/moving_mean

5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
_output_shapes
:0*
dtype0
ข
%batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*6
shared_name'%batch_normalization_4/moving_variance

9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
_output_shapes
:0*
dtype0
ช
#depthwise_conv2d_1/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*4
shared_name%#depthwise_conv2d_1/depthwise_kernel
ฃ
7depthwise_conv2d_1/depthwise_kernel/Read/ReadVariableOpReadVariableOp#depthwise_conv2d_1/depthwise_kernel*&
_output_shapes
:0*
dtype0

batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*+
shared_namebatch_normalization_5/beta

.batch_normalization_5/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_5/beta*
_output_shapes
:0*
dtype0

!batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*2
shared_name#!batch_normalization_5/moving_mean

5batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_5/moving_mean*
_output_shapes
:0*
dtype0
ข
%batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*6
shared_name'%batch_normalization_5/moving_variance

9batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_5/moving_variance*
_output_shapes
:0*
dtype0

conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0 * 
shared_nameconv2d_4/kernel
{
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*&
_output_shapes
:0 *
dtype0

batch_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_6/gamma

/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_6/gamma*
_output_shapes
: *
dtype0

batch_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_6/beta

.batch_normalization_6/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_6/beta*
_output_shapes
: *
dtype0

!batch_normalization_6/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_6/moving_mean

5batch_normalization_6/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_6/moving_mean*
_output_shapes
: *
dtype0
ข
%batch_normalization_6/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_6/moving_variance

9batch_normalization_6/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_6/moving_variance*
_output_shapes
: *
dtype0

conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0* 
shared_nameconv2d_5/kernel
{
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*&
_output_shapes
: 0*
dtype0

batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*+
shared_namebatch_normalization_7/beta

.batch_normalization_7/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_7/beta*
_output_shapes
:0*
dtype0

!batch_normalization_7/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*2
shared_name#!batch_normalization_7/moving_mean

5batch_normalization_7/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_7/moving_mean*
_output_shapes
:0*
dtype0
ข
%batch_normalization_7/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*6
shared_name'%batch_normalization_7/moving_variance

9batch_normalization_7/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_7/moving_variance*
_output_shapes
:0*
dtype0
ช
#depthwise_conv2d_2/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*4
shared_name%#depthwise_conv2d_2/depthwise_kernel
ฃ
7depthwise_conv2d_2/depthwise_kernel/Read/ReadVariableOpReadVariableOp#depthwise_conv2d_2/depthwise_kernel*&
_output_shapes
:0*
dtype0

batch_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*+
shared_namebatch_normalization_8/beta

.batch_normalization_8/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_8/beta*
_output_shapes
:0*
dtype0

!batch_normalization_8/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*2
shared_name#!batch_normalization_8/moving_mean

5batch_normalization_8/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_8/moving_mean*
_output_shapes
:0*
dtype0
ข
%batch_normalization_8/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*6
shared_name'%batch_normalization_8/moving_variance

9batch_normalization_8/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_8/moving_variance*
_output_shapes
:0*
dtype0

conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0@* 
shared_nameconv2d_6/kernel
{
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*&
_output_shapes
:0@*
dtype0

batch_normalization_9/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_9/gamma

/batch_normalization_9/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_9/gamma*
_output_shapes
:@*
dtype0

batch_normalization_9/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_9/beta

.batch_normalization_9/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_9/beta*
_output_shapes
:@*
dtype0

!batch_normalization_9/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_9/moving_mean

5batch_normalization_9/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_9/moving_mean*
_output_shapes
:@*
dtype0
ข
%batch_normalization_9/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_9/moving_variance

9batch_normalization_9/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_9/moving_variance*
_output_shapes
:@*
dtype0

conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@`* 
shared_nameconv2d_7/kernel
{
#conv2d_7/kernel/Read/ReadVariableOpReadVariableOpconv2d_7/kernel*&
_output_shapes
:@`*
dtype0

batch_normalization_10/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*,
shared_namebatch_normalization_10/beta

/batch_normalization_10/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_10/beta*
_output_shapes
:`*
dtype0

"batch_normalization_10/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*3
shared_name$"batch_normalization_10/moving_mean

6batch_normalization_10/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_10/moving_mean*
_output_shapes
:`*
dtype0
ค
&batch_normalization_10/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*7
shared_name(&batch_normalization_10/moving_variance

:batch_normalization_10/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_10/moving_variance*
_output_shapes
:`*
dtype0
ช
#depthwise_conv2d_3/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*4
shared_name%#depthwise_conv2d_3/depthwise_kernel
ฃ
7depthwise_conv2d_3/depthwise_kernel/Read/ReadVariableOpReadVariableOp#depthwise_conv2d_3/depthwise_kernel*&
_output_shapes
:`*
dtype0

batch_normalization_11/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*,
shared_namebatch_normalization_11/beta

/batch_normalization_11/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_11/beta*
_output_shapes
:`*
dtype0

"batch_normalization_11/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*3
shared_name$"batch_normalization_11/moving_mean

6batch_normalization_11/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_11/moving_mean*
_output_shapes
:`*
dtype0
ค
&batch_normalization_11/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*7
shared_name(&batch_normalization_11/moving_variance

:batch_normalization_11/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_11/moving_variance*
_output_shapes
:`*
dtype0

conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:`@* 
shared_nameconv2d_8/kernel
{
#conv2d_8/kernel/Read/ReadVariableOpReadVariableOpconv2d_8/kernel*&
_output_shapes
:`@*
dtype0

batch_normalization_12/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_12/gamma

0batch_normalization_12/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_12/gamma*
_output_shapes
:@*
dtype0

batch_normalization_12/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_12/beta

/batch_normalization_12/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_12/beta*
_output_shapes
:@*
dtype0

"batch_normalization_12/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_12/moving_mean

6batch_normalization_12/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_12/moving_mean*
_output_shapes
:@*
dtype0
ค
&batch_normalization_12/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_12/moving_variance

:batch_normalization_12/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_12/moving_variance*
_output_shapes
:@*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:@*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
า
ConstConst*
_output_shapes
: *
dtype0*
valueB "  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?

Const_1Const*
_output_shapes
:0*
dtype0*ุ
valueฮBห0"ภ  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?

Const_2Const*
_output_shapes
:0*
dtype0*ุ
valueฮBห0"ภ  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?

Const_3Const*
_output_shapes
:0*
dtype0*ุ
valueฮBห0"ภ  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?

Const_4Const*
_output_shapes
:0*
dtype0*ุ
valueฮBห0"ภ  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?

Const_5Const*
_output_shapes
:0*
dtype0*ุ
valueฮBห0"ภ  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?

Const_6Const*
_output_shapes
:0*
dtype0*ุ
valueฮBห0"ภ  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?
ิ
Const_7Const*
_output_shapes
:`*
dtype0*
valueB`"  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?
ิ
Const_8Const*
_output_shapes
:`*
dtype0*
valueB`"  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?

NoOpNoOp
ยด
Const_9Const"/device:CPU:0*
_output_shapes
: *
dtype0*๚ณ
value๏ณB๋ณ Bใณ
๓

layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer_with_weights-6
layer-11
layer_with_weights-7
layer-12
layer-13
layer_with_weights-8
layer-14
layer_with_weights-9
layer-15
layer-16
layer_with_weights-10
layer-17
layer_with_weights-11
layer-18
layer-19
layer_with_weights-12
layer-20
layer_with_weights-13
layer-21
layer_with_weights-14
layer-22
layer_with_weights-15
layer-23
layer-24
layer_with_weights-16
layer-25
layer_with_weights-17
layer-26
layer-27
layer_with_weights-18
layer-28
layer_with_weights-19
layer-29
layer_with_weights-20
layer-30
 layer_with_weights-21
 layer-31
!layer-32
"layer_with_weights-22
"layer-33
#layer_with_weights-23
#layer-34
$layer-35
%layer_with_weights-24
%layer-36
&layer_with_weights-25
&layer-37
'layer-38
(layer-39
)layer-40
*layer_with_weights-26
*layer-41
+trainable_variables
,regularization_losses
-	variables
.	keras_api
/
signatures
 
R
0regularization_losses
1trainable_variables
2	variables
3	keras_api
^

4kernel
5regularization_losses
6trainable_variables
7	variables
8	keras_api

9axis
:beta
;moving_mean
<moving_variance
=regularization_losses
>trainable_variables
?	variables
@	keras_api
R
Aregularization_losses
Btrainable_variables
C	variables
D	keras_api
^

Ekernel
Fregularization_losses
Gtrainable_variables
H	variables
I	keras_api

Jaxis
Kbeta
Lmoving_mean
Mmoving_variance
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api
R
Rregularization_losses
Strainable_variables
T	variables
U	keras_api
h
Vdepthwise_kernel
Wregularization_losses
Xtrainable_variables
Y	variables
Z	keras_api

[axis
\beta
]moving_mean
^moving_variance
_regularization_losses
`trainable_variables
a	variables
b	keras_api
R
cregularization_losses
dtrainable_variables
e	variables
f	keras_api
^

gkernel
hregularization_losses
itrainable_variables
j	variables
k	keras_api

laxis
	mgamma
nbeta
omoving_mean
pmoving_variance
qregularization_losses
rtrainable_variables
s	variables
t	keras_api
R
uregularization_losses
vtrainable_variables
w	variables
x	keras_api
^

ykernel
zregularization_losses
{trainable_variables
|	variables
}	keras_api

~axis
beta
moving_mean
moving_variance
regularization_losses
trainable_variables
	variables
	keras_api
V
regularization_losses
trainable_variables
	variables
	keras_api
m
depthwise_kernel
regularization_losses
trainable_variables
	variables
	keras_api

	axis
	beta
moving_mean
moving_variance
regularization_losses
trainable_variables
	variables
	keras_api
V
regularization_losses
trainable_variables
	variables
	keras_api
c
kernel
regularization_losses
trainable_variables
	variables
	keras_api
?
	?axis

กgamma
	ขbeta
ฃmoving_mean
คmoving_variance
ฅregularization_losses
ฆtrainable_variables
ง	variables
จ	keras_api
c
ฉkernel
ชregularization_losses
ซtrainable_variables
ฌ	variables
ญ	keras_api

	ฎaxis
	ฏbeta
ฐmoving_mean
ฑmoving_variance
ฒregularization_losses
ณtrainable_variables
ด	variables
ต	keras_api
V
ถregularization_losses
ทtrainable_variables
ธ	variables
น	keras_api
m
บdepthwise_kernel
ปregularization_losses
ผtrainable_variables
ฝ	variables
พ	keras_api

	ฟaxis
	ภbeta
มmoving_mean
ยmoving_variance
รregularization_losses
ฤtrainable_variables
ล	variables
ฦ	keras_api
V
วregularization_losses
ศtrainable_variables
ษ	variables
ส	keras_api
c
หkernel
ฬregularization_losses
อtrainable_variables
ฮ	variables
ฯ	keras_api
?
	ะaxis

ัgamma
	าbeta
ำmoving_mean
ิmoving_variance
ีregularization_losses
ึtrainable_variables
ื	variables
ุ	keras_api
c
ูkernel
ฺregularization_losses
?trainable_variables
?	variables
?	keras_api

	?axis
	฿beta
เmoving_mean
แmoving_variance
โregularization_losses
ใtrainable_variables
ไ	variables
ๅ	keras_api
V
ๆregularization_losses
็trainable_variables
่	variables
้	keras_api
m
๊depthwise_kernel
๋regularization_losses
์trainable_variables
ํ	variables
๎	keras_api

	๏axis
	๐beta
๑moving_mean
๒moving_variance
๓regularization_losses
๔trainable_variables
๕	variables
๖	keras_api
V
๗regularization_losses
๘trainable_variables
๙	variables
๚	keras_api
c
๛kernel
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
	axis

gamma
	beta
moving_mean
moving_variance
regularization_losses
trainable_variables
	variables
	keras_api
V
regularization_losses
trainable_variables
	variables
	keras_api
V
regularization_losses
trainable_variables
	variables
	keras_api
V
regularization_losses
trainable_variables
	variables
	keras_api
n
kernel
	bias
regularization_losses
trainable_variables
	variables
	keras_api

40
:1
E2
K3
V4
\5
g6
m7
n8
y9
10
11
12
13
ก14
ข15
ฉ16
ฏ17
บ18
ภ19
ห20
ั21
า22
ู23
฿24
๊25
๐26
๛27
28
29
30
31
 
ํ
40
:1
;2
<3
E4
K5
L6
M7
V8
\9
]10
^11
g12
m13
n14
o15
p16
y17
18
19
20
21
22
23
24
25
ก26
ข27
ฃ28
ค29
ฉ30
ฏ31
ฐ32
ฑ33
บ34
ภ35
ม36
ย37
ห38
ั39
า40
ำ41
ิ42
ู43
฿44
เ45
แ46
๊47
๐48
๑49
๒50
๛51
52
53
54
55
56
57
ฒ
+trainable_variables
,regularization_losses
-	variables
non_trainable_variables
layer_metrics
layers
 layer_regularization_losses
metrics
 
 
 
 
ฒ
0regularization_losses
1trainable_variables
2	variables
?non_trainable_variables
กlayer_metrics
ขlayers
 ฃlayer_regularization_losses
คmetrics
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

40

40
ฒ
5regularization_losses
6trainable_variables
7	variables
ฅnon_trainable_variables
ฆlayer_metrics
งlayers
 จlayer_regularization_losses
ฉmetrics
 
b`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

:0

:0
;1
<2
ฒ
=regularization_losses
>trainable_variables
?	variables
ชnon_trainable_variables
ซlayer_metrics
ฌlayers
 ญlayer_regularization_losses
ฎmetrics
 
 
 
ฒ
Aregularization_losses
Btrainable_variables
C	variables
ฏnon_trainable_variables
ฐlayer_metrics
ฑlayers
 ฒlayer_regularization_losses
ณmetrics
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

E0

E0
ฒ
Fregularization_losses
Gtrainable_variables
H	variables
ดnon_trainable_variables
ตlayer_metrics
ถlayers
 ทlayer_regularization_losses
ธmetrics
 
db
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

K0

K0
L1
M2
ฒ
Nregularization_losses
Otrainable_variables
P	variables
นnon_trainable_variables
บlayer_metrics
ปlayers
 ผlayer_regularization_losses
ฝmetrics
 
 
 
ฒ
Rregularization_losses
Strainable_variables
T	variables
พnon_trainable_variables
ฟlayer_metrics
ภlayers
 มlayer_regularization_losses
ยmetrics
wu
VARIABLE_VALUE!depthwise_conv2d/depthwise_kernel@layer_with_weights-4/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
 

V0

V0
ฒ
Wregularization_losses
Xtrainable_variables
Y	variables
รnon_trainable_variables
ฤlayer_metrics
ลlayers
 ฦlayer_regularization_losses
วmetrics
 
db
VARIABLE_VALUEbatch_normalization_2/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_2/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_2/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

\0

\0
]1
^2
ฒ
_regularization_losses
`trainable_variables
a	variables
ศnon_trainable_variables
ษlayer_metrics
สlayers
 หlayer_regularization_losses
ฬmetrics
 
 
 
ฒ
cregularization_losses
dtrainable_variables
e	variables
อnon_trainable_variables
ฮlayer_metrics
ฯlayers
 ะlayer_regularization_losses
ัmetrics
[Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

g0

g0
ฒ
hregularization_losses
itrainable_variables
j	variables
าnon_trainable_variables
ำlayer_metrics
ิlayers
 ีlayer_regularization_losses
ึmetrics
 
fd
VARIABLE_VALUEbatch_normalization_3/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_3/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_3/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_3/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

m0
n1

m0
n1
o2
p3
ฒ
qregularization_losses
rtrainable_variables
s	variables
ืnon_trainable_variables
ุlayer_metrics
ูlayers
 ฺlayer_regularization_losses
?metrics
 
 
 
ฒ
uregularization_losses
vtrainable_variables
w	variables
?non_trainable_variables
?layer_metrics
?layers
 ฿layer_regularization_losses
เmetrics
[Y
VARIABLE_VALUEconv2d_3/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

y0

y0
ฒ
zregularization_losses
{trainable_variables
|	variables
แnon_trainable_variables
โlayer_metrics
ใlayers
 ไlayer_regularization_losses
ๅmetrics
 
db
VARIABLE_VALUEbatch_normalization_4/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_4/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_4/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
1
2
ต
regularization_losses
trainable_variables
	variables
ๆnon_trainable_variables
็layer_metrics
่layers
 ้layer_regularization_losses
๊metrics
 
 
 
ต
regularization_losses
trainable_variables
	variables
๋non_trainable_variables
์layer_metrics
ํlayers
 ๎layer_regularization_losses
๏metrics
zx
VARIABLE_VALUE#depthwise_conv2d_1/depthwise_kernelAlayer_with_weights-10/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
ต
regularization_losses
trainable_variables
	variables
๐non_trainable_variables
๑layer_metrics
๒layers
 ๓layer_regularization_losses
๔metrics
 
ec
VARIABLE_VALUEbatch_normalization_5/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_5/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_5/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
1
2
ต
regularization_losses
trainable_variables
	variables
๕non_trainable_variables
๖layer_metrics
๗layers
 ๘layer_regularization_losses
๙metrics
 
 
 
ต
regularization_losses
trainable_variables
	variables
๚non_trainable_variables
๛layer_metrics
?layers
 ?layer_regularization_losses
?metrics
\Z
VARIABLE_VALUEconv2d_4/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
ต
regularization_losses
trainable_variables
	variables
?non_trainable_variables
layer_metrics
layers
 layer_regularization_losses
metrics
 
ge
VARIABLE_VALUEbatch_normalization_6/gamma6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_6/beta5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_6/moving_mean<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_6/moving_variance@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

ก0
ข1
 
ก0
ข1
ฃ2
ค3
ต
ฅregularization_losses
ฆtrainable_variables
ง	variables
non_trainable_variables
layer_metrics
layers
 layer_regularization_losses
metrics
\Z
VARIABLE_VALUEconv2d_5/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

ฉ0

ฉ0
ต
ชregularization_losses
ซtrainable_variables
ฌ	variables
non_trainable_variables
layer_metrics
layers
 layer_regularization_losses
metrics
 
ec
VARIABLE_VALUEbatch_normalization_7/beta5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_7/moving_mean<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_7/moving_variance@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

ฏ0

ฏ0
ฐ1
ฑ2
ต
ฒregularization_losses
ณtrainable_variables
ด	variables
non_trainable_variables
layer_metrics
layers
 layer_regularization_losses
metrics
 
 
 
ต
ถregularization_losses
ทtrainable_variables
ธ	variables
non_trainable_variables
layer_metrics
layers
 layer_regularization_losses
metrics
zx
VARIABLE_VALUE#depthwise_conv2d_2/depthwise_kernelAlayer_with_weights-16/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
 

บ0

บ0
ต
ปregularization_losses
ผtrainable_variables
ฝ	variables
non_trainable_variables
layer_metrics
layers
 layer_regularization_losses
metrics
 
ec
VARIABLE_VALUEbatch_normalization_8/beta5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_8/moving_mean<layer_with_weights-17/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_8/moving_variance@layer_with_weights-17/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

ภ0

ภ0
ม1
ย2
ต
รregularization_losses
ฤtrainable_variables
ล	variables
non_trainable_variables
layer_metrics
layers
 ?layer_regularization_losses
กmetrics
 
 
 
ต
วregularization_losses
ศtrainable_variables
ษ	variables
ขnon_trainable_variables
ฃlayer_metrics
คlayers
 ฅlayer_regularization_losses
ฆmetrics
\Z
VARIABLE_VALUEconv2d_6/kernel7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

ห0

ห0
ต
ฬregularization_losses
อtrainable_variables
ฮ	variables
งnon_trainable_variables
จlayer_metrics
ฉlayers
 ชlayer_regularization_losses
ซmetrics
 
ge
VARIABLE_VALUEbatch_normalization_9/gamma6layer_with_weights-19/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_9/beta5layer_with_weights-19/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_9/moving_mean<layer_with_weights-19/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_9/moving_variance@layer_with_weights-19/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

ั0
า1
 
ั0
า1
ำ2
ิ3
ต
ีregularization_losses
ึtrainable_variables
ื	variables
ฌnon_trainable_variables
ญlayer_metrics
ฎlayers
 ฏlayer_regularization_losses
ฐmetrics
\Z
VARIABLE_VALUEconv2d_7/kernel7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

ู0

ู0
ต
ฺregularization_losses
?trainable_variables
?	variables
ฑnon_trainable_variables
ฒlayer_metrics
ณlayers
 ดlayer_regularization_losses
ตmetrics
 
fd
VARIABLE_VALUEbatch_normalization_10/beta5layer_with_weights-21/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_10/moving_mean<layer_with_weights-21/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_10/moving_variance@layer_with_weights-21/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

฿0

฿0
เ1
แ2
ต
โregularization_losses
ใtrainable_variables
ไ	variables
ถnon_trainable_variables
ทlayer_metrics
ธlayers
 นlayer_regularization_losses
บmetrics
 
 
 
ต
ๆregularization_losses
็trainable_variables
่	variables
ปnon_trainable_variables
ผlayer_metrics
ฝlayers
 พlayer_regularization_losses
ฟmetrics
zx
VARIABLE_VALUE#depthwise_conv2d_3/depthwise_kernelAlayer_with_weights-22/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
 

๊0

๊0
ต
๋regularization_losses
์trainable_variables
ํ	variables
ภnon_trainable_variables
มlayer_metrics
ยlayers
 รlayer_regularization_losses
ฤmetrics
 
fd
VARIABLE_VALUEbatch_normalization_11/beta5layer_with_weights-23/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_11/moving_mean<layer_with_weights-23/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_11/moving_variance@layer_with_weights-23/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

๐0

๐0
๑1
๒2
ต
๓regularization_losses
๔trainable_variables
๕	variables
ลnon_trainable_variables
ฦlayer_metrics
วlayers
 ศlayer_regularization_losses
ษmetrics
 
 
 
ต
๗regularization_losses
๘trainable_variables
๙	variables
สnon_trainable_variables
หlayer_metrics
ฬlayers
 อlayer_regularization_losses
ฮmetrics
\Z
VARIABLE_VALUEconv2d_8/kernel7layer_with_weights-24/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

๛0

๛0
ต
?regularization_losses
?trainable_variables
?	variables
ฯnon_trainable_variables
ะlayer_metrics
ัlayers
 าlayer_regularization_losses
ำmetrics
 
hf
VARIABLE_VALUEbatch_normalization_12/gamma6layer_with_weights-25/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_12/beta5layer_with_weights-25/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_12/moving_mean<layer_with_weights-25/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_12/moving_variance@layer_with_weights-25/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
 
0
1
2
3
ต
regularization_losses
trainable_variables
	variables
ิnon_trainable_variables
ีlayer_metrics
ึlayers
 ืlayer_regularization_losses
ุmetrics
 
 
 
ต
regularization_losses
trainable_variables
	variables
ูnon_trainable_variables
ฺlayer_metrics
?layers
 ?layer_regularization_losses
?metrics
 
 
 
ต
regularization_losses
trainable_variables
	variables
?non_trainable_variables
฿layer_metrics
เlayers
 แlayer_regularization_losses
โmetrics
 
 
 
ต
regularization_losses
trainable_variables
	variables
ใnon_trainable_variables
ไlayer_metrics
ๅlayers
 ๆlayer_regularization_losses
็metrics
YW
VARIABLE_VALUEdense/kernel7layer_with_weights-26/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUE
dense/bias5layer_with_weights-26/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
ต
regularization_losses
trainable_variables
	variables
่non_trainable_variables
้layer_metrics
๊layers
 ๋layer_regularization_losses
์metrics
ุ
;0
<1
L2
M3
]4
^5
o6
p7
8
9
10
11
ฃ12
ค13
ฐ14
ฑ15
ม16
ย17
ำ18
ิ19
เ20
แ21
๑22
๒23
24
25
 
ฦ
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41
 
 
 
 
 
 
 
 
 
 
 
 

;0
<1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

L0
M1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

]0
^1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

o0
p1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

0
1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

0
1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

ฃ0
ค1
 
 
 
 
 
 
 
 
 

ฐ0
ฑ1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

ม0
ย1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

ำ0
ิ1
 
 
 
 
 
 
 
 
 

เ0
แ1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

๑0
๒1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

0
1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
p
serving_default_input_1Placeholder*"
_output_shapes
:1(*
dtype0*
shape:1(
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d/kernelbatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceConstconv2d_1/kernelbatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceConst_1!depthwise_conv2d/depthwise_kernelbatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceConst_2conv2d_2/kernelbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv2d_3/kernelbatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_varianceConst_3#depthwise_conv2d_1/depthwise_kernelbatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_varianceConst_4conv2d_4/kernelbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_varianceconv2d_5/kernelbatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_varianceConst_5#depthwise_conv2d_2/depthwise_kernelbatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_varianceConst_6conv2d_6/kernelbatch_normalization_9/gammabatch_normalization_9/beta!batch_normalization_9/moving_mean%batch_normalization_9/moving_varianceconv2d_7/kernelbatch_normalization_10/beta"batch_normalization_10/moving_mean&batch_normalization_10/moving_varianceConst_7#depthwise_conv2d_3/depthwise_kernelbatch_normalization_11/beta"batch_normalization_11/moving_mean&batch_normalization_11/moving_varianceConst_8conv2d_8/kernelbatch_normalization_12/gammabatch_normalization_12/beta"batch_normalization_12/moving_mean&batch_normalization_12/moving_variancedense/kernel
dense/bias*O
TinH
F2D*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*\
_read_only_resource_inputs>
<:	 !"#$%&')*+,./012345689:;=>?@ABC*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference_signature_wrapper_3923
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ฦ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp5depthwise_conv2d/depthwise_kernel/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp.batch_normalization_4/beta/Read/ReadVariableOp5batch_normalization_4/moving_mean/Read/ReadVariableOp9batch_normalization_4/moving_variance/Read/ReadVariableOp7depthwise_conv2d_1/depthwise_kernel/Read/ReadVariableOp.batch_normalization_5/beta/Read/ReadVariableOp5batch_normalization_5/moving_mean/Read/ReadVariableOp9batch_normalization_5/moving_variance/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp/batch_normalization_6/gamma/Read/ReadVariableOp.batch_normalization_6/beta/Read/ReadVariableOp5batch_normalization_6/moving_mean/Read/ReadVariableOp9batch_normalization_6/moving_variance/Read/ReadVariableOp#conv2d_5/kernel/Read/ReadVariableOp.batch_normalization_7/beta/Read/ReadVariableOp5batch_normalization_7/moving_mean/Read/ReadVariableOp9batch_normalization_7/moving_variance/Read/ReadVariableOp7depthwise_conv2d_2/depthwise_kernel/Read/ReadVariableOp.batch_normalization_8/beta/Read/ReadVariableOp5batch_normalization_8/moving_mean/Read/ReadVariableOp9batch_normalization_8/moving_variance/Read/ReadVariableOp#conv2d_6/kernel/Read/ReadVariableOp/batch_normalization_9/gamma/Read/ReadVariableOp.batch_normalization_9/beta/Read/ReadVariableOp5batch_normalization_9/moving_mean/Read/ReadVariableOp9batch_normalization_9/moving_variance/Read/ReadVariableOp#conv2d_7/kernel/Read/ReadVariableOp/batch_normalization_10/beta/Read/ReadVariableOp6batch_normalization_10/moving_mean/Read/ReadVariableOp:batch_normalization_10/moving_variance/Read/ReadVariableOp7depthwise_conv2d_3/depthwise_kernel/Read/ReadVariableOp/batch_normalization_11/beta/Read/ReadVariableOp6batch_normalization_11/moving_mean/Read/ReadVariableOp:batch_normalization_11/moving_variance/Read/ReadVariableOp#conv2d_8/kernel/Read/ReadVariableOp0batch_normalization_12/gamma/Read/ReadVariableOp/batch_normalization_12/beta/Read/ReadVariableOp6batch_normalization_12/moving_mean/Read/ReadVariableOp:batch_normalization_12/moving_variance/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpConst_9*G
Tin@
>2<*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *&
f!R
__inference__traced_save_7069
ท
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelbatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d_1/kernelbatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variance!depthwise_conv2d/depthwise_kernelbatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv2d_2/kernelbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv2d_3/kernelbatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variance#depthwise_conv2d_1/depthwise_kernelbatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_varianceconv2d_4/kernelbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_varianceconv2d_5/kernelbatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_variance#depthwise_conv2d_2/depthwise_kernelbatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_varianceconv2d_6/kernelbatch_normalization_9/gammabatch_normalization_9/beta!batch_normalization_9/moving_mean%batch_normalization_9/moving_varianceconv2d_7/kernelbatch_normalization_10/beta"batch_normalization_10/moving_mean&batch_normalization_10/moving_variance#depthwise_conv2d_3/depthwise_kernelbatch_normalization_11/beta"batch_normalization_11/moving_mean&batch_normalization_11/moving_varianceconv2d_8/kernelbatch_normalization_12/gammabatch_normalization_12/beta"batch_normalization_12/moving_mean&batch_normalization_12/moving_variancedense/kernel
dense/bias*F
Tin?
=2;*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__traced_restore_7253จ)
๐	

4__inference_batch_normalization_8_layer_call_fn_6314

inputs
batch_normalization_8_beta%
!batch_normalization_8_moving_mean)
%batch_normalization_8_moving_variance
batch_normalization_8_const
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_8_beta!batch_normalization_8_moving_mean%batch_normalization_8_moving_variancebatch_normalization_8_const*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????0*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_17972
StatefulPartitionedCallจ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:+???????????????????????????0::::022
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs: 

_output_shapes
:0
ญ
[
?__inference_re_lu_layer_call_and_return_conditional_losses_5342

inputs
identityP
Relu6Relu6inputs*
T0*&
_output_shapes
: 2
Relu6f
IdentityIdentityRelu6:activations:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*%
_input_shapes
: :N J
&
_output_shapes
: 
 
_user_specified_nameinputs
ะ

`
A__inference_dropout_layer_call_and_return_conditional_losses_3327

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Constj
dropout/MulMulinputsdropout/Const:output:0*
T0*
_output_shapes

:@2
dropout/Mulo
dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2
dropout/Shapeซ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
_output_shapes

:@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *อฬL>2
dropout/GreaterEqual/yต
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*
_output_shapes

:@2
dropout/GreaterEqualv
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:@2
dropout/Castq
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*
_output_shapes

:@2
dropout/Mul_1\
IdentityIdentitydropout/Mul_1:z:0*
T0*
_output_shapes

:@2

Identity"
identityIdentity:output:0*
_input_shapes

:@:F B

_output_shapes

:@
 
_user_specified_nameinputs
ง

O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2426

inputs-
)readvariableop_batch_normalization_2_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_2_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance0
,fusedbatchnormv3_batch_normalization_2_const
identityขAssignNewValueขAssignNewValue_1
ReadVariableOpReadVariableOp)readvariableop_batch_normalization_2_beta*
_output_shapes
:0*
dtype02
ReadVariableOpภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ใ
FusedBatchNormV3FusedBatchNormV3inputs,fusedbatchnormv3_batch_normalization_2_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
ื#<2
FusedBatchNormV3ฑ
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_2_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_2/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueว
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_2/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:0::::02 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:N J
&
_output_shapes
:0
 
_user_specified_nameinputs: 

_output_shapes
:0

B
&__inference_re_lu_3_layer_call_fn_5848

inputs
identityพ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_re_lu_3_layer_call_and_return_conditional_losses_26502
PartitionedCallk
IdentityIdentityPartitionedCall:output:0*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*%
_input_shapes
:0:N J
&
_output_shapes
:0
 
_user_specified_nameinputs
ษ

B__inference_conv2d_3_layer_call_and_return_conditional_losses_5728

inputs)
%conv2d_readvariableop_conv2d_3_kernel
identity
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_3_kernel*&
_output_shapes
: 0*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:0*
paddingSAME*
strides
2
Conv2Db
IdentityIdentityConv2D:output:0*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*)
_input_shapes
: ::N J
&
_output_shapes
: 
 
_user_specified_nameinputs


O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1183

inputs-
)readvariableop_batch_normalization_2_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_2_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance0
,fusedbatchnormv3_batch_normalization_2_const
identityขAssignNewValueขAssignNewValue_1
ReadVariableOpReadVariableOp)readvariableop_batch_normalization_2_beta*
_output_shapes
:0*
dtype02
ReadVariableOpภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputs,fusedbatchnormv3_batch_normalization_2_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
ื#<2
FusedBatchNormV3ฑ
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_2_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_2/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueว
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_2/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1ฆ
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:+???????????????????????????0::::02 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs: 

_output_shapes
:0
๋

O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_6047

inputs.
*readvariableop_batch_normalization_6_gamma/
+readvariableop_1_batch_normalization_6_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_6_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance
identityขAssignNewValueขAssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_6_gamma*
_output_shapes
: *
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_6_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1ภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_6_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1๊
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o:*
exponential_avg_factor%
ื#<2
FusedBatchNormV3ฑ
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_6_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_6/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueว
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_6/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1ฆ
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs


P__inference_batch_normalization_12_layer_call_and_return_conditional_losses_6717

inputs/
+readvariableop_batch_normalization_12_gamma0
,readvariableop_1_batch_normalization_12_betaF
Bfusedbatchnormv3_readvariableop_batch_normalization_12_moving_meanL
Hfusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance
identityขAssignNewValueขAssignNewValue_1
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_12_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_12_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1ม
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_12_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpห
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ฯ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
ื#<2
FusedBatchNormV3ณ
AssignNewValueAssignVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_12_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*U
_classK
IGloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_12/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueษ
AssignNewValue_1AssignVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*[
_classQ
OMloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_12/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*5
_input_shapes$
":@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:N J
&
_output_shapes
:@
 
_user_specified_nameinputs
	

4__inference_batch_normalization_5_layer_call_fn_5900

inputs
batch_normalization_5_beta%
!batch_normalization_5_moving_mean)
%batch_normalization_5_moving_variance
batch_normalization_5_const
identityขStatefulPartitionedCall๐
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_5_beta!batch_normalization_5_moving_mean%batch_normalization_5_moving_variancebatch_normalization_5_const*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_26992
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:0::::022
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:0
 
_user_specified_nameinputs: 

_output_shapes
:0


O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_5917

inputs-
)readvariableop_batch_normalization_5_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_5_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance0
,fusedbatchnormv3_batch_normalization_5_const
identityขAssignNewValueขAssignNewValue_1
ReadVariableOpReadVariableOp)readvariableop_batch_normalization_5_beta*
_output_shapes
:0*
dtype02
ReadVariableOpภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_5_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputs,fusedbatchnormv3_batch_normalization_5_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
ื#<2
FusedBatchNormV3ฑ
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_5_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_5/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueว
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_5/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1ฆ
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:+???????????????????????????0::::02 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs: 

_output_shapes
:0
๋
่
P__inference_batch_normalization_12_layer_call_and_return_conditional_losses_6735

inputs/
+readvariableop_batch_normalization_12_gamma0
,readvariableop_1_batch_normalization_12_betaF
Bfusedbatchnormv3_readvariableop_batch_normalization_12_moving_meanL
Hfusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance
identity
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_12_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_12_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1ม
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_12_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpห
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ม
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3g
IdentityIdentityFusedBatchNormV3:y:0*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*5
_input_shapes$
":@:::::N J
&
_output_shapes
:@
 
_user_specified_nameinputs
ผ-
฿
+__inference_functional_1_layer_call_fn_4494
input_1
conv2d_kernel
batch_normalization_beta#
batch_normalization_moving_mean'
#batch_normalization_moving_variance
batch_normalization_const
conv2d_1_kernel
batch_normalization_1_beta%
!batch_normalization_1_moving_mean)
%batch_normalization_1_moving_variance
batch_normalization_1_const%
!depthwise_conv2d_depthwise_kernel
batch_normalization_2_beta%
!batch_normalization_2_moving_mean)
%batch_normalization_2_moving_variance
batch_normalization_2_const
conv2d_2_kernel
batch_normalization_3_gamma
batch_normalization_3_beta%
!batch_normalization_3_moving_mean)
%batch_normalization_3_moving_variance
conv2d_3_kernel
batch_normalization_4_beta%
!batch_normalization_4_moving_mean)
%batch_normalization_4_moving_variance
batch_normalization_4_const'
#depthwise_conv2d_1_depthwise_kernel
batch_normalization_5_beta%
!batch_normalization_5_moving_mean)
%batch_normalization_5_moving_variance
batch_normalization_5_const
conv2d_4_kernel
batch_normalization_6_gamma
batch_normalization_6_beta%
!batch_normalization_6_moving_mean)
%batch_normalization_6_moving_variance
conv2d_5_kernel
batch_normalization_7_beta%
!batch_normalization_7_moving_mean)
%batch_normalization_7_moving_variance
batch_normalization_7_const'
#depthwise_conv2d_2_depthwise_kernel
batch_normalization_8_beta%
!batch_normalization_8_moving_mean)
%batch_normalization_8_moving_variance
batch_normalization_8_const
conv2d_6_kernel
batch_normalization_9_gamma
batch_normalization_9_beta%
!batch_normalization_9_moving_mean)
%batch_normalization_9_moving_variance
conv2d_7_kernel
batch_normalization_10_beta&
"batch_normalization_10_moving_mean*
&batch_normalization_10_moving_variance 
batch_normalization_10_const'
#depthwise_conv2d_3_depthwise_kernel
batch_normalization_11_beta&
"batch_normalization_11_moving_mean*
&batch_normalization_11_moving_variance 
batch_normalization_11_const
conv2d_8_kernel 
batch_normalization_12_gamma
batch_normalization_12_beta&
"batch_normalization_12_moving_mean*
&batch_normalization_12_moving_variance
dense_kernel

dense_bias
identityขStatefulPartitionedCallง
StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_kernelbatch_normalization_betabatch_normalization_moving_mean#batch_normalization_moving_variancebatch_normalization_constconv2d_1_kernelbatch_normalization_1_beta!batch_normalization_1_moving_mean%batch_normalization_1_moving_variancebatch_normalization_1_const!depthwise_conv2d_depthwise_kernelbatch_normalization_2_beta!batch_normalization_2_moving_mean%batch_normalization_2_moving_variancebatch_normalization_2_constconv2d_2_kernelbatch_normalization_3_gammabatch_normalization_3_beta!batch_normalization_3_moving_mean%batch_normalization_3_moving_varianceconv2d_3_kernelbatch_normalization_4_beta!batch_normalization_4_moving_mean%batch_normalization_4_moving_variancebatch_normalization_4_const#depthwise_conv2d_1_depthwise_kernelbatch_normalization_5_beta!batch_normalization_5_moving_mean%batch_normalization_5_moving_variancebatch_normalization_5_constconv2d_4_kernelbatch_normalization_6_gammabatch_normalization_6_beta!batch_normalization_6_moving_mean%batch_normalization_6_moving_varianceconv2d_5_kernelbatch_normalization_7_beta!batch_normalization_7_moving_mean%batch_normalization_7_moving_variancebatch_normalization_7_const#depthwise_conv2d_2_depthwise_kernelbatch_normalization_8_beta!batch_normalization_8_moving_mean%batch_normalization_8_moving_variancebatch_normalization_8_constconv2d_6_kernelbatch_normalization_9_gammabatch_normalization_9_beta!batch_normalization_9_moving_mean%batch_normalization_9_moving_varianceconv2d_7_kernelbatch_normalization_10_beta"batch_normalization_10_moving_mean&batch_normalization_10_moving_variancebatch_normalization_10_const#depthwise_conv2d_3_depthwise_kernelbatch_normalization_11_beta"batch_normalization_11_moving_mean&batch_normalization_11_moving_variancebatch_normalization_11_constconv2d_8_kernelbatch_normalization_12_gammabatch_normalization_12_beta"batch_normalization_12_moving_mean&batch_normalization_12_moving_variancedense_kernel
dense_bias*O
TinH
F2D*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*B
_read_only_resource_inputs$
"  !$%)*./03489=>?BC*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_functional_1_layer_call_and_return_conditional_losses_35952
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*ส
_input_shapesธ
ต:?????????1(::::: :::::0:::::0::::::::::0:::::0::::::::::0:::::0::::::::::`:::::`:::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????1(
!
_user_specified_name	input_1: 

_output_shapes
: : 


_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0: (

_output_shapes
:0: -

_output_shapes
:0: 7

_output_shapes
:`: <

_output_shapes
:`
๗
ใ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_6182

inputs-
)readvariableop_batch_normalization_7_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_7_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance0
,fusedbatchnormv3_batch_normalization_7_const
identity
ReadVariableOpReadVariableOp)readvariableop_batch_normalization_7_beta*
_output_shapes
:0*
dtype02
ReadVariableOpภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_7_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1๐
FusedBatchNormV3FusedBatchNormV3inputs,fusedbatchnormv3_batch_normalization_7_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:+???????????????????????????0::::0:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs: 

_output_shapes
:0

ใ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2862

inputs-
)readvariableop_batch_normalization_7_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_7_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance0
,fusedbatchnormv3_batch_normalization_7_const
identity
ReadVariableOpReadVariableOp)readvariableop_batch_normalization_7_beta*
_output_shapes
:0*
dtype02
ReadVariableOpภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_7_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ี
FusedBatchNormV3FusedBatchNormV3inputs,fusedbatchnormv3_batch_normalization_7_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:0:0:0:0:0:*
epsilon%o:*
is_training( 2
FusedBatchNormV3g
IdentityIdentityFusedBatchNormV3:y:0*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:0::::0:N J
&
_output_shapes
:0
 
_user_specified_nameinputs: 

_output_shapes
:0

ใ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5560

inputs-
)readvariableop_batch_normalization_2_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_2_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance0
,fusedbatchnormv3_batch_normalization_2_const
identity
ReadVariableOpReadVariableOp)readvariableop_batch_normalization_2_beta*
_output_shapes
:0*
dtype02
ReadVariableOpภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ี
FusedBatchNormV3FusedBatchNormV3inputs,fusedbatchnormv3_batch_normalization_2_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:0:0:0:0:0:*
epsilon%o:*
is_training( 2
FusedBatchNormV3g
IdentityIdentityFusedBatchNormV3:y:0*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:0::::0:N J
&
_output_shapes
:0
 
_user_specified_nameinputs: 

_output_shapes
:0
ฆ
ญ
?__inference_dense_layer_call_and_return_conditional_losses_6856

inputs&
"matmul_readvariableop_dense_kernel%
!biasadd_readvariableop_dense_bias
identity
MatMul/ReadVariableOpReadVariableOp"matmul_readvariableop_dense_kernel*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOpj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
MatMul
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_dense_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2	
BiasAdd[
IdentityIdentityBiasAdd:output:0*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*%
_input_shapes
:@:::F B

_output_shapes

:@
 
_user_specified_nameinputs
ฯ
ใ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_5691

inputs.
*readvariableop_batch_normalization_3_gamma/
+readvariableop_1_batch_normalization_3_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_3_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_3_gamma*
_output_shapes
: *
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_3_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1ภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? :::::i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
ษ

B__inference_conv2d_7_layer_call_and_return_conditional_losses_6452

inputs)
%conv2d_readvariableop_conv2d_7_kernel
identity
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_7_kernel*&
_output_shapes
:@`*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:`*
paddingSAME*
strides
2
Conv2Db
IdentityIdentityConv2D:output:0*
T0*&
_output_shapes
:`2

Identity"
identityIdentity:output:0*)
_input_shapes
:@::N J
&
_output_shapes
:@
 
_user_specified_nameinputs
๘
ู
M__inference_batch_normalization_layer_call_and_return_conditional_losses_2278

inputs+
'readvariableop_batch_normalization_betaC
?fusedbatchnormv3_readvariableop_batch_normalization_moving_meanI
Efusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance.
*fusedbatchnormv3_batch_normalization_const
identity
ReadVariableOpReadVariableOp'readvariableop_batch_normalization_beta*
_output_shapes
: *
dtype02
ReadVariableOpพ
FusedBatchNormV3/ReadVariableOpReadVariableOp?fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpศ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpEfusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ำ
FusedBatchNormV3FusedBatchNormV3inputs*fusedbatchnormv3_batch_normalization_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3g
IdentityIdentityFusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*7
_input_shapes&
$: :::: :N J
&
_output_shapes
: 
 
_user_specified_nameinputs: 

_output_shapes
: 
๎	

4__inference_batch_normalization_4_layer_call_fn_5829

inputs
batch_normalization_4_beta%
!batch_normalization_4_moving_mean)
%batch_normalization_4_moving_variance
batch_normalization_4_const
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_4_beta!batch_normalization_4_moving_mean%batch_normalization_4_moving_variancebatch_normalization_4_const*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_13632
StatefulPartitionedCallจ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:+???????????????????????????0::::022
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs: 

_output_shapes
:0
ด

P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_3086

inputs.
*readvariableop_batch_normalization_10_betaF
Bfusedbatchnormv3_readvariableop_batch_normalization_10_moving_meanL
Hfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance1
-fusedbatchnormv3_batch_normalization_10_const
identityขAssignNewValueขAssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_10_beta*
_output_shapes
:`*
dtype02
ReadVariableOpม
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_10_moving_mean*
_output_shapes
:`*
dtype02!
FusedBatchNormV3/ReadVariableOpห
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance*
_output_shapes
:`*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ไ
FusedBatchNormV3FusedBatchNormV3inputs-fusedbatchnormv3_batch_normalization_10_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:`:`:`:`:`:*
epsilon%o:*
exponential_avg_factor%
ื#<2
FusedBatchNormV3ณ
AssignNewValueAssignVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_10_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*U
_classK
IGloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_10/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueษ
AssignNewValue_1AssignVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*[
_classQ
OMloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_10/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*&
_output_shapes
:`2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:`::::`2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:N J
&
_output_shapes
:`
 
_user_specified_nameinputs: 

_output_shapes
:`

ใ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_5882

inputs-
)readvariableop_batch_normalization_5_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_5_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance0
,fusedbatchnormv3_batch_normalization_5_const
identity
ReadVariableOpReadVariableOp)readvariableop_batch_normalization_5_beta*
_output_shapes
:0*
dtype02
ReadVariableOpภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_5_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ี
FusedBatchNormV3FusedBatchNormV3inputs,fusedbatchnormv3_batch_normalization_5_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:0:0:0:0:0:*
epsilon%o:*
is_training( 2
FusedBatchNormV3g
IdentityIdentityFusedBatchNormV3:y:0*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:0::::0:N J
&
_output_shapes
:0
 
_user_specified_nameinputs: 

_output_shapes
:0
ด

P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_3164

inputs.
*readvariableop_batch_normalization_11_betaF
Bfusedbatchnormv3_readvariableop_batch_normalization_11_moving_meanL
Hfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance1
-fusedbatchnormv3_batch_normalization_11_const
identityขAssignNewValueขAssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_11_beta*
_output_shapes
:`*
dtype02
ReadVariableOpม
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_11_moving_mean*
_output_shapes
:`*
dtype02!
FusedBatchNormV3/ReadVariableOpห
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance*
_output_shapes
:`*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ไ
FusedBatchNormV3FusedBatchNormV3inputs-fusedbatchnormv3_batch_normalization_11_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:`:`:`:`:`:*
epsilon%o:*
exponential_avg_factor%
ื#<2
FusedBatchNormV3ณ
AssignNewValueAssignVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_11_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*U
_classK
IGloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_11/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueษ
AssignNewValue_1AssignVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*[
_classQ
OMloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_11/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*&
_output_shapes
:`2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:`::::`2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:N J
&
_output_shapes
:`
 
_user_specified_nameinputs: 

_output_shapes
:`
ฏ
]
A__inference_re_lu_8_layer_call_and_return_conditional_losses_3210

inputs
identityP
Relu6Relu6inputs*
T0*&
_output_shapes
:`2
Relu6f
IdentityIdentityRelu6:activations:0*
T0*&
_output_shapes
:`2

Identity"
identityIdentity:output:0*%
_input_shapes
:`:N J
&
_output_shapes
:`
 
_user_specified_nameinputs
๗
ใ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5820

inputs-
)readvariableop_batch_normalization_4_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_4_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance0
,fusedbatchnormv3_batch_normalization_4_const
identity
ReadVariableOpReadVariableOp)readvariableop_batch_normalization_4_beta*
_output_shapes
:0*
dtype02
ReadVariableOpภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1๐
FusedBatchNormV3FusedBatchNormV3inputs,fusedbatchnormv3_batch_normalization_4_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:+???????????????????????????0::::0:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs: 

_output_shapes
:0

@
$__inference_re_lu_layer_call_fn_5347

inputs
identityผ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_re_lu_layer_call_and_return_conditional_losses_23072
PartitionedCallk
IdentityIdentityPartitionedCall:output:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*%
_input_shapes
: :N J
&
_output_shapes
: 
 
_user_specified_nameinputs
ฯ
ใ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_6065

inputs.
*readvariableop_batch_normalization_6_gamma/
+readvariableop_1_batch_normalization_6_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_6_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_6_gamma*
_output_shapes
: *
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_6_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1ภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_6_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? :::::i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
ง

O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_6227

inputs-
)readvariableop_batch_normalization_8_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_8_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance0
,fusedbatchnormv3_batch_normalization_8_const
identityขAssignNewValueขAssignNewValue_1
ReadVariableOpReadVariableOp)readvariableop_batch_normalization_8_beta*
_output_shapes
:0*
dtype02
ReadVariableOpภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ใ
FusedBatchNormV3FusedBatchNormV3inputs,fusedbatchnormv3_batch_normalization_8_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
ื#<2
FusedBatchNormV3ฑ
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_8_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_8/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueว
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_8/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:0::::02 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:N J
&
_output_shapes
:0
 
_user_specified_nameinputs: 

_output_shapes
:0

N
"__inference_add_layer_call_fn_5721
inputs_0
inputs_1
identityว
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *F
fAR?
=__inference_add_layer_call_and_return_conditional_losses_25622
PartitionedCallk
IdentityIdentityPartitionedCall:output:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*7
_input_shapes&
$: : :P L
&
_output_shapes
: 
"
_user_specified_name
inputs/0:PL
&
_output_shapes
: 
"
_user_specified_name
inputs/1
ษ

B__inference_conv2d_8_layer_call_and_return_conditional_losses_6693

inputs)
%conv2d_readvariableop_conv2d_8_kernel
identity
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_8_kernel*&
_output_shapes
:`@*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingSAME*
strides
2
Conv2Db
IdentityIdentityConv2D:output:0*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*)
_input_shapes
:`::N J
&
_output_shapes
:`
 
_user_specified_nameinputs
?

P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_2065

inputs.
*readvariableop_batch_normalization_11_betaF
Bfusedbatchnormv3_readvariableop_batch_normalization_11_moving_meanL
Hfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance1
-fusedbatchnormv3_batch_normalization_11_const
identityขAssignNewValueขAssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_11_beta*
_output_shapes
:`*
dtype02
ReadVariableOpม
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_11_moving_mean*
_output_shapes
:`*
dtype02!
FusedBatchNormV3/ReadVariableOpห
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance*
_output_shapes
:`*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputs-fusedbatchnormv3_batch_normalization_11_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????`:`:`:`:`:*
epsilon%o:*
exponential_avg_factor%
ื#<2
FusedBatchNormV3ณ
AssignNewValueAssignVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_11_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*U
_classK
IGloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_11/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueษ
AssignNewValue_1AssignVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*[
_classQ
OMloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_11/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1ฆ
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+???????????????????????????`2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:+???????????????????????????`::::`2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+???????????????????????????`
 
_user_specified_nameinputs: 

_output_shapes
:`
ุ
่
P__inference_batch_normalization_12_layer_call_and_return_conditional_losses_6789

inputs/
+readvariableop_batch_normalization_12_gamma0
,readvariableop_1_batch_normalization_12_betaF
Bfusedbatchnormv3_readvariableop_batch_normalization_12_moving_meanL
Hfusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance
identity
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_12_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_12_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1ม
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_12_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpห
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@:::::i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
ง

O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_6113

inputs-
)readvariableop_batch_normalization_7_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_7_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance0
,fusedbatchnormv3_batch_normalization_7_const
identityขAssignNewValueขAssignNewValue_1
ReadVariableOpReadVariableOp)readvariableop_batch_normalization_7_beta*
_output_shapes
:0*
dtype02
ReadVariableOpภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_7_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ใ
FusedBatchNormV3FusedBatchNormV3inputs,fusedbatchnormv3_batch_normalization_7_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
ื#<2
FusedBatchNormV3ฑ
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_7_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_7/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueว
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_7/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:0::::02 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:N J
&
_output_shapes
:0
 
_user_specified_nameinputs: 

_output_shapes
:0
	

5__inference_batch_normalization_10_layer_call_fn_6553

inputs
batch_normalization_10_beta&
"batch_normalization_10_moving_mean*
&batch_normalization_10_moving_variance 
batch_normalization_10_const
identityขStatefulPartitionedCall๓
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_10_beta"batch_normalization_10_moving_mean&batch_normalization_10_moving_variancebatch_normalization_10_const*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:`*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_30862
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:`2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:`::::`22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:`
 
_user_specified_nameinputs: 

_output_shapes
:`
ฤ

@__inference_conv2d_layer_call_and_return_conditional_losses_5227

inputs'
#conv2d_readvariableop_conv2d_kernel
identity
Conv2D/ReadVariableOpReadVariableOp#conv2d_readvariableop_conv2d_kernel*&
_output_shapes
:( *
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
2
Conv2Db
IdentityIdentityConv2D:output:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*)
_input_shapes
:1(::N J
&
_output_shapes
:1(
 
_user_specified_nameinputs
ฏ
]
A__inference_re_lu_3_layer_call_and_return_conditional_losses_5843

inputs
identityP
Relu6Relu6inputs*
T0*&
_output_shapes
:02
Relu6f
IdentityIdentityRelu6:activations:0*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*%
_input_shapes
:0:N J
&
_output_shapes
:0
 
_user_specified_nameinputs
ธ
n
R__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_2197

inputs
identity
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs

่
P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_3181

inputs.
*readvariableop_batch_normalization_11_betaF
Bfusedbatchnormv3_readvariableop_batch_normalization_11_moving_meanL
Hfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance1
-fusedbatchnormv3_batch_normalization_11_const
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_11_beta*
_output_shapes
:`*
dtype02
ReadVariableOpม
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_11_moving_mean*
_output_shapes
:`*
dtype02!
FusedBatchNormV3/ReadVariableOpห
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance*
_output_shapes
:`*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ึ
FusedBatchNormV3FusedBatchNormV3inputs-fusedbatchnormv3_batch_normalization_11_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:`:`:`:`:`:*
epsilon%o:*
is_training( 2
FusedBatchNormV3g
IdentityIdentityFusedBatchNormV3:y:0*
T0*&
_output_shapes
:`2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:`::::`:N J
&
_output_shapes
:`
 
_user_specified_nameinputs: 

_output_shapes
:`
ง

O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2845

inputs-
)readvariableop_batch_normalization_7_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_7_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance0
,fusedbatchnormv3_batch_normalization_7_const
identityขAssignNewValueขAssignNewValue_1
ReadVariableOpReadVariableOp)readvariableop_batch_normalization_7_beta*
_output_shapes
:0*
dtype02
ReadVariableOpภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_7_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ใ
FusedBatchNormV3FusedBatchNormV3inputs,fusedbatchnormv3_batch_normalization_7_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
ื#<2
FusedBatchNormV3ฑ
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_7_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_7/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueว
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_7/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:0::::02 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:N J
&
_output_shapes
:0
 
_user_specified_nameinputs: 

_output_shapes
:0

?
M__inference_batch_normalization_layer_call_and_return_conditional_losses_5302

inputs+
'readvariableop_batch_normalization_betaC
?fusedbatchnormv3_readvariableop_batch_normalization_moving_meanI
Efusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance.
*fusedbatchnormv3_batch_normalization_const
identityขAssignNewValueขAssignNewValue_1
ReadVariableOpReadVariableOp'readvariableop_batch_normalization_beta*
_output_shapes
: *
dtype02
ReadVariableOpพ
FusedBatchNormV3/ReadVariableOpReadVariableOp?fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpศ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpEfusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1แ
FusedBatchNormV3FusedBatchNormV3inputs*fusedbatchnormv3_batch_normalization_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
exponential_avg_factor%
ื#<2
FusedBatchNormV3ญ
AssignNewValueAssignVariableOp?fusedbatchnormv3_readvariableop_batch_normalization_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueร
AssignNewValue_1AssignVariableOpEfusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*X
_classN
LJloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*7
_input_shapes&
$: :::: 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:N J
&
_output_shapes
: 
 
_user_specified_nameinputs: 

_output_shapes
: 
๗
ใ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1209

inputs-
)readvariableop_batch_normalization_2_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_2_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance0
,fusedbatchnormv3_batch_normalization_2_const
identity
ReadVariableOpReadVariableOp)readvariableop_batch_normalization_2_beta*
_output_shapes
:0*
dtype02
ReadVariableOpภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1๐
FusedBatchNormV3FusedBatchNormV3inputs,fusedbatchnormv3_batch_normalization_2_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:+???????????????????????????0::::0:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs: 

_output_shapes
:0

่
P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_6606

inputs.
*readvariableop_batch_normalization_11_betaF
Bfusedbatchnormv3_readvariableop_batch_normalization_11_moving_meanL
Hfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance1
-fusedbatchnormv3_batch_normalization_11_const
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_11_beta*
_output_shapes
:`*
dtype02
ReadVariableOpม
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_11_moving_mean*
_output_shapes
:`*
dtype02!
FusedBatchNormV3/ReadVariableOpห
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance*
_output_shapes
:`*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ึ
FusedBatchNormV3FusedBatchNormV3inputs-fusedbatchnormv3_batch_normalization_11_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:`:`:`:`:`:*
epsilon%o:*
is_training( 2
FusedBatchNormV3g
IdentityIdentityFusedBatchNormV3:y:0*
T0*&
_output_shapes
:`2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:`::::`:N J
&
_output_shapes
:`
 
_user_specified_nameinputs: 

_output_shapes
:`
๋

O__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1862

inputs.
*readvariableop_batch_normalization_9_gamma/
+readvariableop_1_batch_normalization_9_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_9_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance
identityขAssignNewValueขAssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_9_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_9_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1ภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_9_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1๊
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
ื#<2
FusedBatchNormV3ฑ
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_9_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_9/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueว
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_9/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1ฆ
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs

่
P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_6658

inputs.
*readvariableop_batch_normalization_11_betaF
Bfusedbatchnormv3_readvariableop_batch_normalization_11_moving_meanL
Hfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance1
-fusedbatchnormv3_batch_normalization_11_const
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_11_beta*
_output_shapes
:`*
dtype02
ReadVariableOpม
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_11_moving_mean*
_output_shapes
:`*
dtype02!
FusedBatchNormV3/ReadVariableOpห
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance*
_output_shapes
:`*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1๑
FusedBatchNormV3FusedBatchNormV3inputs-fusedbatchnormv3_batch_normalization_11_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????`:`:`:`:`:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????`2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:+???????????????????????????`::::`:i e
A
_output_shapes/
-:+???????????????????????????`
 
_user_specified_nameinputs: 

_output_shapes
:`
ุ
่
P__inference_batch_normalization_12_layer_call_and_return_conditional_losses_2183

inputs/
+readvariableop_batch_normalization_12_gamma0
,readvariableop_1_batch_normalization_12_betaF
Bfusedbatchnormv3_readvariableop_batch_normalization_12_moving_meanL
Hfusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance
identity
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_12_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_12_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1ม
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_12_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpห
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@:::::i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
ษ

B__inference_conv2d_5_layer_call_and_return_conditional_losses_6090

inputs)
%conv2d_readvariableop_conv2d_5_kernel
identity
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_5_kernel*&
_output_shapes
: 0*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:0*
paddingSAME*
strides
2
Conv2Db
IdentityIdentityConv2D:output:0*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*)
_input_shapes
: ::N J
&
_output_shapes
: 
 
_user_specified_nameinputs

ใ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5768

inputs-
)readvariableop_batch_normalization_4_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_4_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance0
,fusedbatchnormv3_batch_normalization_4_const
identity
ReadVariableOpReadVariableOp)readvariableop_batch_normalization_4_beta*
_output_shapes
:0*
dtype02
ReadVariableOpภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ี
FusedBatchNormV3FusedBatchNormV3inputs,fusedbatchnormv3_batch_normalization_4_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:0:0:0:0:0:*
epsilon%o:*
is_training( 2
FusedBatchNormV3g
IdentityIdentityFusedBatchNormV3:y:0*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:0::::0:N J
&
_output_shapes
:0
 
_user_specified_nameinputs: 

_output_shapes
:0
๋

O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_5673

inputs.
*readvariableop_batch_normalization_3_gamma/
+readvariableop_1_batch_normalization_3_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_3_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance
identityขAssignNewValueขAssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_3_gamma*
_output_shapes
: *
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_3_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1ภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1๊
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o:*
exponential_avg_factor%
ื#<2
FusedBatchNormV3ฑ
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_3_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_3/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueว
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_3/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1ฆ
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs

ใ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2365

inputs-
)readvariableop_batch_normalization_1_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_1_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance0
,fusedbatchnormv3_batch_normalization_1_const
identity
ReadVariableOpReadVariableOp)readvariableop_batch_normalization_1_beta*
_output_shapes
:0*
dtype02
ReadVariableOpภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ี
FusedBatchNormV3FusedBatchNormV3inputs,fusedbatchnormv3_batch_normalization_1_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:0:0:0:0:0:*
epsilon%o:*
is_training( 2
FusedBatchNormV3g
IdentityIdentityFusedBatchNormV3:y:0*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:0::::0:N J
&
_output_shapes
:0
 
_user_specified_nameinputs: 

_output_shapes
:0

่
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_3103

inputs.
*readvariableop_batch_normalization_10_betaF
Bfusedbatchnormv3_readvariableop_batch_normalization_10_moving_meanL
Hfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance1
-fusedbatchnormv3_batch_normalization_10_const
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_10_beta*
_output_shapes
:`*
dtype02
ReadVariableOpม
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_10_moving_mean*
_output_shapes
:`*
dtype02!
FusedBatchNormV3/ReadVariableOpห
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance*
_output_shapes
:`*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ึ
FusedBatchNormV3FusedBatchNormV3inputs-fusedbatchnormv3_batch_normalization_10_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:`:`:`:`:`:*
epsilon%o:*
is_training( 2
FusedBatchNormV3g
IdentityIdentityFusedBatchNormV3:y:0*
T0*&
_output_shapes
:`2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:`::::`:N J
&
_output_shapes
:`
 
_user_specified_nameinputs: 

_output_shapes
:`
ง

O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2682

inputs-
)readvariableop_batch_normalization_5_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_5_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance0
,fusedbatchnormv3_batch_normalization_5_const
identityขAssignNewValueขAssignNewValue_1
ReadVariableOpReadVariableOp)readvariableop_batch_normalization_5_beta*
_output_shapes
:0*
dtype02
ReadVariableOpภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_5_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ใ
FusedBatchNormV3FusedBatchNormV3inputs,fusedbatchnormv3_batch_normalization_5_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
ื#<2
FusedBatchNormV3ฑ
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_5_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_5/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueว
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_5/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:0::::02 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:N J
&
_output_shapes
:0
 
_user_specified_nameinputs: 

_output_shapes
:0
โ๘
#
 __inference__traced_restore_7253
file_prefix"
assignvariableop_conv2d_kernel/
+assignvariableop_1_batch_normalization_beta6
2assignvariableop_2_batch_normalization_moving_mean:
6assignvariableop_3_batch_normalization_moving_variance&
"assignvariableop_4_conv2d_1_kernel1
-assignvariableop_5_batch_normalization_1_beta8
4assignvariableop_6_batch_normalization_1_moving_mean<
8assignvariableop_7_batch_normalization_1_moving_variance8
4assignvariableop_8_depthwise_conv2d_depthwise_kernel1
-assignvariableop_9_batch_normalization_2_beta9
5assignvariableop_10_batch_normalization_2_moving_mean=
9assignvariableop_11_batch_normalization_2_moving_variance'
#assignvariableop_12_conv2d_2_kernel3
/assignvariableop_13_batch_normalization_3_gamma2
.assignvariableop_14_batch_normalization_3_beta9
5assignvariableop_15_batch_normalization_3_moving_mean=
9assignvariableop_16_batch_normalization_3_moving_variance'
#assignvariableop_17_conv2d_3_kernel2
.assignvariableop_18_batch_normalization_4_beta9
5assignvariableop_19_batch_normalization_4_moving_mean=
9assignvariableop_20_batch_normalization_4_moving_variance;
7assignvariableop_21_depthwise_conv2d_1_depthwise_kernel2
.assignvariableop_22_batch_normalization_5_beta9
5assignvariableop_23_batch_normalization_5_moving_mean=
9assignvariableop_24_batch_normalization_5_moving_variance'
#assignvariableop_25_conv2d_4_kernel3
/assignvariableop_26_batch_normalization_6_gamma2
.assignvariableop_27_batch_normalization_6_beta9
5assignvariableop_28_batch_normalization_6_moving_mean=
9assignvariableop_29_batch_normalization_6_moving_variance'
#assignvariableop_30_conv2d_5_kernel2
.assignvariableop_31_batch_normalization_7_beta9
5assignvariableop_32_batch_normalization_7_moving_mean=
9assignvariableop_33_batch_normalization_7_moving_variance;
7assignvariableop_34_depthwise_conv2d_2_depthwise_kernel2
.assignvariableop_35_batch_normalization_8_beta9
5assignvariableop_36_batch_normalization_8_moving_mean=
9assignvariableop_37_batch_normalization_8_moving_variance'
#assignvariableop_38_conv2d_6_kernel3
/assignvariableop_39_batch_normalization_9_gamma2
.assignvariableop_40_batch_normalization_9_beta9
5assignvariableop_41_batch_normalization_9_moving_mean=
9assignvariableop_42_batch_normalization_9_moving_variance'
#assignvariableop_43_conv2d_7_kernel3
/assignvariableop_44_batch_normalization_10_beta:
6assignvariableop_45_batch_normalization_10_moving_mean>
:assignvariableop_46_batch_normalization_10_moving_variance;
7assignvariableop_47_depthwise_conv2d_3_depthwise_kernel3
/assignvariableop_48_batch_normalization_11_beta:
6assignvariableop_49_batch_normalization_11_moving_mean>
:assignvariableop_50_batch_normalization_11_moving_variance'
#assignvariableop_51_conv2d_8_kernel4
0assignvariableop_52_batch_normalization_12_gamma3
/assignvariableop_53_batch_normalization_12_beta:
6assignvariableop_54_batch_normalization_12_moving_mean>
:assignvariableop_55_batch_normalization_12_moving_variance$
 assignvariableop_56_dense_kernel"
assignvariableop_57_dense_bias
identity_59ขAssignVariableOpขAssignVariableOp_1ขAssignVariableOp_10ขAssignVariableOp_11ขAssignVariableOp_12ขAssignVariableOp_13ขAssignVariableOp_14ขAssignVariableOp_15ขAssignVariableOp_16ขAssignVariableOp_17ขAssignVariableOp_18ขAssignVariableOp_19ขAssignVariableOp_2ขAssignVariableOp_20ขAssignVariableOp_21ขAssignVariableOp_22ขAssignVariableOp_23ขAssignVariableOp_24ขAssignVariableOp_25ขAssignVariableOp_26ขAssignVariableOp_27ขAssignVariableOp_28ขAssignVariableOp_29ขAssignVariableOp_3ขAssignVariableOp_30ขAssignVariableOp_31ขAssignVariableOp_32ขAssignVariableOp_33ขAssignVariableOp_34ขAssignVariableOp_35ขAssignVariableOp_36ขAssignVariableOp_37ขAssignVariableOp_38ขAssignVariableOp_39ขAssignVariableOp_4ขAssignVariableOp_40ขAssignVariableOp_41ขAssignVariableOp_42ขAssignVariableOp_43ขAssignVariableOp_44ขAssignVariableOp_45ขAssignVariableOp_46ขAssignVariableOp_47ขAssignVariableOp_48ขAssignVariableOp_49ขAssignVariableOp_5ขAssignVariableOp_50ขAssignVariableOp_51ขAssignVariableOp_52ขAssignVariableOp_53ขAssignVariableOp_54ขAssignVariableOp_55ขAssignVariableOp_56ขAssignVariableOp_57ขAssignVariableOp_6ขAssignVariableOp_7ขAssignVariableOp_8ขAssignVariableOp_9บ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:;*
dtype0*ฦ
valueผBน;B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-4/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-10/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-16/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-17/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-17/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-19/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-19/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-19/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-21/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-21/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-22/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-23/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-23/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-23/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-24/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-25/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-25/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-25/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-25/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-26/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-26/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:;*
dtype0*
valueB~;B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesี
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes๏
์:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*I
dtypes?
=2;2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1ฐ
AssignVariableOp_1AssignVariableOp+assignvariableop_1_batch_normalization_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2ท
AssignVariableOp_2AssignVariableOp2assignvariableop_2_batch_normalization_moving_meanIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3ป
AssignVariableOp_3AssignVariableOp6assignvariableop_3_batch_normalization_moving_varianceIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4ง
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5ฒ
AssignVariableOp_5AssignVariableOp-assignvariableop_5_batch_normalization_1_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6น
AssignVariableOp_6AssignVariableOp4assignvariableop_6_batch_normalization_1_moving_meanIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7ฝ
AssignVariableOp_7AssignVariableOp8assignvariableop_7_batch_normalization_1_moving_varianceIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8น
AssignVariableOp_8AssignVariableOp4assignvariableop_8_depthwise_conv2d_depthwise_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9ฒ
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_2_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10ฝ
AssignVariableOp_10AssignVariableOp5assignvariableop_10_batch_normalization_2_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11ม
AssignVariableOp_11AssignVariableOp9assignvariableop_11_batch_normalization_2_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12ซ
AssignVariableOp_12AssignVariableOp#assignvariableop_12_conv2d_2_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13ท
AssignVariableOp_13AssignVariableOp/assignvariableop_13_batch_normalization_3_gammaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14ถ
AssignVariableOp_14AssignVariableOp.assignvariableop_14_batch_normalization_3_betaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15ฝ
AssignVariableOp_15AssignVariableOp5assignvariableop_15_batch_normalization_3_moving_meanIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16ม
AssignVariableOp_16AssignVariableOp9assignvariableop_16_batch_normalization_3_moving_varianceIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17ซ
AssignVariableOp_17AssignVariableOp#assignvariableop_17_conv2d_3_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18ถ
AssignVariableOp_18AssignVariableOp.assignvariableop_18_batch_normalization_4_betaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19ฝ
AssignVariableOp_19AssignVariableOp5assignvariableop_19_batch_normalization_4_moving_meanIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20ม
AssignVariableOp_20AssignVariableOp9assignvariableop_20_batch_normalization_4_moving_varianceIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21ฟ
AssignVariableOp_21AssignVariableOp7assignvariableop_21_depthwise_conv2d_1_depthwise_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22ถ
AssignVariableOp_22AssignVariableOp.assignvariableop_22_batch_normalization_5_betaIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23ฝ
AssignVariableOp_23AssignVariableOp5assignvariableop_23_batch_normalization_5_moving_meanIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24ม
AssignVariableOp_24AssignVariableOp9assignvariableop_24_batch_normalization_5_moving_varianceIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25ซ
AssignVariableOp_25AssignVariableOp#assignvariableop_25_conv2d_4_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26ท
AssignVariableOp_26AssignVariableOp/assignvariableop_26_batch_normalization_6_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27ถ
AssignVariableOp_27AssignVariableOp.assignvariableop_27_batch_normalization_6_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28ฝ
AssignVariableOp_28AssignVariableOp5assignvariableop_28_batch_normalization_6_moving_meanIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29ม
AssignVariableOp_29AssignVariableOp9assignvariableop_29_batch_normalization_6_moving_varianceIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30ซ
AssignVariableOp_30AssignVariableOp#assignvariableop_30_conv2d_5_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31ถ
AssignVariableOp_31AssignVariableOp.assignvariableop_31_batch_normalization_7_betaIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32ฝ
AssignVariableOp_32AssignVariableOp5assignvariableop_32_batch_normalization_7_moving_meanIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33ม
AssignVariableOp_33AssignVariableOp9assignvariableop_33_batch_normalization_7_moving_varianceIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34ฟ
AssignVariableOp_34AssignVariableOp7assignvariableop_34_depthwise_conv2d_2_depthwise_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35ถ
AssignVariableOp_35AssignVariableOp.assignvariableop_35_batch_normalization_8_betaIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36ฝ
AssignVariableOp_36AssignVariableOp5assignvariableop_36_batch_normalization_8_moving_meanIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37ม
AssignVariableOp_37AssignVariableOp9assignvariableop_37_batch_normalization_8_moving_varianceIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38ซ
AssignVariableOp_38AssignVariableOp#assignvariableop_38_conv2d_6_kernelIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39ท
AssignVariableOp_39AssignVariableOp/assignvariableop_39_batch_normalization_9_gammaIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40ถ
AssignVariableOp_40AssignVariableOp.assignvariableop_40_batch_normalization_9_betaIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41ฝ
AssignVariableOp_41AssignVariableOp5assignvariableop_41_batch_normalization_9_moving_meanIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42ม
AssignVariableOp_42AssignVariableOp9assignvariableop_42_batch_normalization_9_moving_varianceIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43ซ
AssignVariableOp_43AssignVariableOp#assignvariableop_43_conv2d_7_kernelIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44ท
AssignVariableOp_44AssignVariableOp/assignvariableop_44_batch_normalization_10_betaIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45พ
AssignVariableOp_45AssignVariableOp6assignvariableop_45_batch_normalization_10_moving_meanIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46ย
AssignVariableOp_46AssignVariableOp:assignvariableop_46_batch_normalization_10_moving_varianceIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47ฟ
AssignVariableOp_47AssignVariableOp7assignvariableop_47_depthwise_conv2d_3_depthwise_kernelIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48ท
AssignVariableOp_48AssignVariableOp/assignvariableop_48_batch_normalization_11_betaIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49พ
AssignVariableOp_49AssignVariableOp6assignvariableop_49_batch_normalization_11_moving_meanIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50ย
AssignVariableOp_50AssignVariableOp:assignvariableop_50_batch_normalization_11_moving_varianceIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51ซ
AssignVariableOp_51AssignVariableOp#assignvariableop_51_conv2d_8_kernelIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52ธ
AssignVariableOp_52AssignVariableOp0assignvariableop_52_batch_normalization_12_gammaIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53ท
AssignVariableOp_53AssignVariableOp/assignvariableop_53_batch_normalization_12_betaIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54พ
AssignVariableOp_54AssignVariableOp6assignvariableop_54_batch_normalization_12_moving_meanIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55ย
AssignVariableOp_55AssignVariableOp:assignvariableop_55_batch_normalization_12_moving_varianceIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56จ
AssignVariableOp_56AssignVariableOp assignvariableop_56_dense_kernelIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57ฆ
AssignVariableOp_57AssignVariableOpassignvariableop_57_dense_biasIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_579
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpฺ

Identity_58Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_58อ

Identity_59IdentityIdentity_58:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_59"#
identity_59Identity_59:output:0*?
_input_shapesํ
๊: ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
๐	

4__inference_batch_normalization_7_layer_call_fn_6200

inputs
batch_normalization_7_beta%
!batch_normalization_7_moving_mean)
%batch_normalization_7_moving_variance
batch_normalization_7_const
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_7_beta!batch_normalization_7_moving_mean%batch_normalization_7_moving_variancebatch_normalization_7_const*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????0*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_16832
StatefulPartitionedCallจ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:+???????????????????????????0::::022
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs: 

_output_shapes
:0
๎	

4__inference_batch_normalization_8_layer_call_fn_6305

inputs
batch_normalization_8_beta%
!batch_normalization_8_moving_mean)
%batch_normalization_8_moving_variance
batch_normalization_8_const
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_8_beta!batch_normalization_8_moving_mean%batch_normalization_8_moving_variancebatch_normalization_8_const*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_17712
StatefulPartitionedCallจ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:+???????????????????????????0::::022
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs: 

_output_shapes
:0
	
ญ
L__inference_depthwise_conv2d_2_layer_call_and_return_conditional_losses_1712

inputs@
<depthwise_readvariableop_depthwise_conv2d_2_depthwise_kernel
identityน
depthwise/ReadVariableOpReadVariableOp<depthwise_readvariableop_depthwise_conv2d_2_depthwise_kernel*&
_output_shapes
:0*
dtype02
depthwise/ReadVariableOp{
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      0      2
depthwise/Shape
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
depthwise/dilation_rateอ
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????0*
paddingSAME*
strides
2
	depthwise
IdentityIdentitydepthwise:output:0*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+???????????????????????????0::i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
๗
ใ
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_6296

inputs-
)readvariableop_batch_normalization_8_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_8_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance0
,fusedbatchnormv3_batch_normalization_8_const
identity
ReadVariableOpReadVariableOp)readvariableop_batch_normalization_8_beta*
_output_shapes
:0*
dtype02
ReadVariableOpภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1๐
FusedBatchNormV3FusedBatchNormV3inputs,fusedbatchnormv3_batch_normalization_8_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:+???????????????????????????0::::0:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs: 

_output_shapes
:0
๗
ใ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5394

inputs-
)readvariableop_batch_normalization_1_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_1_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance0
,fusedbatchnormv3_batch_normalization_1_const
identity
ReadVariableOpReadVariableOp)readvariableop_batch_normalization_1_beta*
_output_shapes
:0*
dtype02
ReadVariableOpภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1๐
FusedBatchNormV3FusedBatchNormV3inputs,fusedbatchnormv3_batch_normalization_1_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:+???????????????????????????0::::0:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs: 

_output_shapes
:0
ํ

1__inference_depthwise_conv2d_3_layer_call_fn_2010

inputs'
#depthwise_conv2d_3_depthwise_kernel
identityขStatefulPartitionedCallฅ
StatefulPartitionedCallStatefulPartitionedCallinputs#depthwise_conv2d_3_depthwise_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????`*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_depthwise_conv2d_3_layer_call_and_return_conditional_losses_20062
StatefulPartitionedCallจ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????`2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+???????????????????????????`:22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????`
 
_user_specified_nameinputs
ึ-
฿
+__inference_functional_1_layer_call_fn_4566
input_1
conv2d_kernel
batch_normalization_beta#
batch_normalization_moving_mean'
#batch_normalization_moving_variance
batch_normalization_const
conv2d_1_kernel
batch_normalization_1_beta%
!batch_normalization_1_moving_mean)
%batch_normalization_1_moving_variance
batch_normalization_1_const%
!depthwise_conv2d_depthwise_kernel
batch_normalization_2_beta%
!batch_normalization_2_moving_mean)
%batch_normalization_2_moving_variance
batch_normalization_2_const
conv2d_2_kernel
batch_normalization_3_gamma
batch_normalization_3_beta%
!batch_normalization_3_moving_mean)
%batch_normalization_3_moving_variance
conv2d_3_kernel
batch_normalization_4_beta%
!batch_normalization_4_moving_mean)
%batch_normalization_4_moving_variance
batch_normalization_4_const'
#depthwise_conv2d_1_depthwise_kernel
batch_normalization_5_beta%
!batch_normalization_5_moving_mean)
%batch_normalization_5_moving_variance
batch_normalization_5_const
conv2d_4_kernel
batch_normalization_6_gamma
batch_normalization_6_beta%
!batch_normalization_6_moving_mean)
%batch_normalization_6_moving_variance
conv2d_5_kernel
batch_normalization_7_beta%
!batch_normalization_7_moving_mean)
%batch_normalization_7_moving_variance
batch_normalization_7_const'
#depthwise_conv2d_2_depthwise_kernel
batch_normalization_8_beta%
!batch_normalization_8_moving_mean)
%batch_normalization_8_moving_variance
batch_normalization_8_const
conv2d_6_kernel
batch_normalization_9_gamma
batch_normalization_9_beta%
!batch_normalization_9_moving_mean)
%batch_normalization_9_moving_variance
conv2d_7_kernel
batch_normalization_10_beta&
"batch_normalization_10_moving_mean*
&batch_normalization_10_moving_variance 
batch_normalization_10_const'
#depthwise_conv2d_3_depthwise_kernel
batch_normalization_11_beta&
"batch_normalization_11_moving_mean*
&batch_normalization_11_moving_variance 
batch_normalization_11_const
conv2d_8_kernel 
batch_normalization_12_gamma
batch_normalization_12_beta&
"batch_normalization_12_moving_mean*
&batch_normalization_12_moving_variance
dense_kernel

dense_bias
identityขStatefulPartitionedCallม
StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_kernelbatch_normalization_betabatch_normalization_moving_mean#batch_normalization_moving_variancebatch_normalization_constconv2d_1_kernelbatch_normalization_1_beta!batch_normalization_1_moving_mean%batch_normalization_1_moving_variancebatch_normalization_1_const!depthwise_conv2d_depthwise_kernelbatch_normalization_2_beta!batch_normalization_2_moving_mean%batch_normalization_2_moving_variancebatch_normalization_2_constconv2d_2_kernelbatch_normalization_3_gammabatch_normalization_3_beta!batch_normalization_3_moving_mean%batch_normalization_3_moving_varianceconv2d_3_kernelbatch_normalization_4_beta!batch_normalization_4_moving_mean%batch_normalization_4_moving_variancebatch_normalization_4_const#depthwise_conv2d_1_depthwise_kernelbatch_normalization_5_beta!batch_normalization_5_moving_mean%batch_normalization_5_moving_variancebatch_normalization_5_constconv2d_4_kernelbatch_normalization_6_gammabatch_normalization_6_beta!batch_normalization_6_moving_mean%batch_normalization_6_moving_varianceconv2d_5_kernelbatch_normalization_7_beta!batch_normalization_7_moving_mean%batch_normalization_7_moving_variancebatch_normalization_7_const#depthwise_conv2d_2_depthwise_kernelbatch_normalization_8_beta!batch_normalization_8_moving_mean%batch_normalization_8_moving_variancebatch_normalization_8_constconv2d_6_kernelbatch_normalization_9_gammabatch_normalization_9_beta!batch_normalization_9_moving_mean%batch_normalization_9_moving_varianceconv2d_7_kernelbatch_normalization_10_beta"batch_normalization_10_moving_mean&batch_normalization_10_moving_variancebatch_normalization_10_const#depthwise_conv2d_3_depthwise_kernelbatch_normalization_11_beta"batch_normalization_11_moving_mean&batch_normalization_11_moving_variancebatch_normalization_11_constconv2d_8_kernelbatch_normalization_12_gammabatch_normalization_12_beta"batch_normalization_12_moving_mean&batch_normalization_12_moving_variancedense_kernel
dense_bias*O
TinH
F2D*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*\
_read_only_resource_inputs>
<:	 !"#$%&')*+,./012345689:;=>?@ABC*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_functional_1_layer_call_and_return_conditional_losses_37792
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*ส
_input_shapesธ
ต:?????????1(::::: :::::0:::::0::::::::::0:::::0::::::::::0:::::0::::::::::`:::::`:::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????1(
!
_user_specified_name	input_1: 

_output_shapes
: : 


_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0: (

_output_shapes
:0: -

_output_shapes
:0: 7

_output_shapes
:`: <

_output_shapes
:`
ฯ
ใ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1301

inputs.
*readvariableop_batch_normalization_3_gamma/
+readvariableop_1_batch_normalization_3_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_3_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_3_gamma*
_output_shapes
: *
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_3_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1ภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? :::::i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
z
?
__inference__traced_save_7069
file_prefix,
(savev2_conv2d_kernel_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop@
<savev2_depthwise_conv2d_depthwise_kernel_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop:
6savev2_batch_normalization_3_gamma_read_readvariableop9
5savev2_batch_normalization_3_beta_read_readvariableop@
<savev2_batch_normalization_3_moving_mean_read_readvariableopD
@savev2_batch_normalization_3_moving_variance_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop9
5savev2_batch_normalization_4_beta_read_readvariableop@
<savev2_batch_normalization_4_moving_mean_read_readvariableopD
@savev2_batch_normalization_4_moving_variance_read_readvariableopB
>savev2_depthwise_conv2d_1_depthwise_kernel_read_readvariableop9
5savev2_batch_normalization_5_beta_read_readvariableop@
<savev2_batch_normalization_5_moving_mean_read_readvariableopD
@savev2_batch_normalization_5_moving_variance_read_readvariableop.
*savev2_conv2d_4_kernel_read_readvariableop:
6savev2_batch_normalization_6_gamma_read_readvariableop9
5savev2_batch_normalization_6_beta_read_readvariableop@
<savev2_batch_normalization_6_moving_mean_read_readvariableopD
@savev2_batch_normalization_6_moving_variance_read_readvariableop.
*savev2_conv2d_5_kernel_read_readvariableop9
5savev2_batch_normalization_7_beta_read_readvariableop@
<savev2_batch_normalization_7_moving_mean_read_readvariableopD
@savev2_batch_normalization_7_moving_variance_read_readvariableopB
>savev2_depthwise_conv2d_2_depthwise_kernel_read_readvariableop9
5savev2_batch_normalization_8_beta_read_readvariableop@
<savev2_batch_normalization_8_moving_mean_read_readvariableopD
@savev2_batch_normalization_8_moving_variance_read_readvariableop.
*savev2_conv2d_6_kernel_read_readvariableop:
6savev2_batch_normalization_9_gamma_read_readvariableop9
5savev2_batch_normalization_9_beta_read_readvariableop@
<savev2_batch_normalization_9_moving_mean_read_readvariableopD
@savev2_batch_normalization_9_moving_variance_read_readvariableop.
*savev2_conv2d_7_kernel_read_readvariableop:
6savev2_batch_normalization_10_beta_read_readvariableopA
=savev2_batch_normalization_10_moving_mean_read_readvariableopE
Asavev2_batch_normalization_10_moving_variance_read_readvariableopB
>savev2_depthwise_conv2d_3_depthwise_kernel_read_readvariableop:
6savev2_batch_normalization_11_beta_read_readvariableopA
=savev2_batch_normalization_11_moving_mean_read_readvariableopE
Asavev2_batch_normalization_11_moving_variance_read_readvariableop.
*savev2_conv2d_8_kernel_read_readvariableop;
7savev2_batch_normalization_12_gamma_read_readvariableop:
6savev2_batch_normalization_12_beta_read_readvariableopA
=savev2_batch_normalization_12_moving_mean_read_readvariableopE
Asavev2_batch_normalization_12_moving_variance_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop
savev2_const_9

identity_1ขMergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_1ba2c2a815bc4273b973bcb616ae6f74/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardฆ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameด
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:;*
dtype0*ฦ
valueผBน;B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-4/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-10/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-16/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-17/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-17/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-19/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-19/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-19/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-21/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-21/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-22/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-23/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-23/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-23/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-24/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-25/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-25/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-25/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-25/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-26/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-26/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:;*
dtype0*
valueB~;B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop<savev2_depthwise_conv2d_depthwise_kernel_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop5savev2_batch_normalization_4_beta_read_readvariableop<savev2_batch_normalization_4_moving_mean_read_readvariableop@savev2_batch_normalization_4_moving_variance_read_readvariableop>savev2_depthwise_conv2d_1_depthwise_kernel_read_readvariableop5savev2_batch_normalization_5_beta_read_readvariableop<savev2_batch_normalization_5_moving_mean_read_readvariableop@savev2_batch_normalization_5_moving_variance_read_readvariableop*savev2_conv2d_4_kernel_read_readvariableop6savev2_batch_normalization_6_gamma_read_readvariableop5savev2_batch_normalization_6_beta_read_readvariableop<savev2_batch_normalization_6_moving_mean_read_readvariableop@savev2_batch_normalization_6_moving_variance_read_readvariableop*savev2_conv2d_5_kernel_read_readvariableop5savev2_batch_normalization_7_beta_read_readvariableop<savev2_batch_normalization_7_moving_mean_read_readvariableop@savev2_batch_normalization_7_moving_variance_read_readvariableop>savev2_depthwise_conv2d_2_depthwise_kernel_read_readvariableop5savev2_batch_normalization_8_beta_read_readvariableop<savev2_batch_normalization_8_moving_mean_read_readvariableop@savev2_batch_normalization_8_moving_variance_read_readvariableop*savev2_conv2d_6_kernel_read_readvariableop6savev2_batch_normalization_9_gamma_read_readvariableop5savev2_batch_normalization_9_beta_read_readvariableop<savev2_batch_normalization_9_moving_mean_read_readvariableop@savev2_batch_normalization_9_moving_variance_read_readvariableop*savev2_conv2d_7_kernel_read_readvariableop6savev2_batch_normalization_10_beta_read_readvariableop=savev2_batch_normalization_10_moving_mean_read_readvariableopAsavev2_batch_normalization_10_moving_variance_read_readvariableop>savev2_depthwise_conv2d_3_depthwise_kernel_read_readvariableop6savev2_batch_normalization_11_beta_read_readvariableop=savev2_batch_normalization_11_moving_mean_read_readvariableopAsavev2_batch_normalization_11_moving_variance_read_readvariableop*savev2_conv2d_8_kernel_read_readvariableop7savev2_batch_normalization_12_gamma_read_readvariableop6savev2_batch_normalization_12_beta_read_readvariableop=savev2_batch_normalization_12_moving_mean_read_readvariableopAsavev2_batch_normalization_12_moving_variance_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableopsavev2_const_9"/device:CPU:0*
_output_shapes
 *I
dtypes?
=2;2
SaveV2บ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesก
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: :( : : : : 0:0:0:0:0:0:0:0:0 : : : : : 0:0:0:0:0:0:0:0:0 : : : : : 0:0:0:0:0:0:0:0:0@:@:@:@:@:@`:`:`:`:`:`:`:`:`@:@:@:@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:( : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: 0: 

_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0:,	(
&
_output_shapes
:0: 


_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0:,(
&
_output_shapes
:0 : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: 0: 

_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0:,(
&
_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0:,(
&
_output_shapes
:0 : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: 0:  

_output_shapes
:0: !

_output_shapes
:0: "

_output_shapes
:0:,#(
&
_output_shapes
:0: $

_output_shapes
:0: %

_output_shapes
:0: &

_output_shapes
:0:,'(
&
_output_shapes
:0@: (

_output_shapes
:@: )

_output_shapes
:@: *

_output_shapes
:@: +

_output_shapes
:@:,,(
&
_output_shapes
:@`: -

_output_shapes
:`: .

_output_shapes
:`: /

_output_shapes
:`:,0(
&
_output_shapes
:`: 1

_output_shapes
:`: 2

_output_shapes
:`: 3

_output_shapes
:`:,4(
&
_output_shapes
:`@: 5

_output_shapes
:@: 6

_output_shapes
:@: 7

_output_shapes
:@: 8

_output_shapes
:@:$9 

_output_shapes

:@: :

_output_shapes
::;

_output_shapes
: 
ด

P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_6589

inputs.
*readvariableop_batch_normalization_11_betaF
Bfusedbatchnormv3_readvariableop_batch_normalization_11_moving_meanL
Hfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance1
-fusedbatchnormv3_batch_normalization_11_const
identityขAssignNewValueขAssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_11_beta*
_output_shapes
:`*
dtype02
ReadVariableOpม
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_11_moving_mean*
_output_shapes
:`*
dtype02!
FusedBatchNormV3/ReadVariableOpห
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance*
_output_shapes
:`*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ไ
FusedBatchNormV3FusedBatchNormV3inputs-fusedbatchnormv3_batch_normalization_11_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:`:`:`:`:`:*
epsilon%o:*
exponential_avg_factor%
ื#<2
FusedBatchNormV3ณ
AssignNewValueAssignVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_11_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*U
_classK
IGloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_11/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueษ
AssignNewValue_1AssignVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*[
_classQ
OMloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_11/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*&
_output_shapes
:`2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:`::::`2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:N J
&
_output_shapes
:`
 
_user_specified_nameinputs: 

_output_shapes
:`
ฤ
u
'__inference_conv2d_2_layer_call_fn_5601

inputs
conv2d_2_kernel
identityขStatefulPartitionedCall์
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_2_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_24872
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*)
_input_shapes
:0:22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:0
 
_user_specified_nameinputs
ฏ
]
A__inference_re_lu_5_layer_call_and_return_conditional_losses_2891

inputs
identityP
Relu6Relu6inputs*
T0*&
_output_shapes
:02
Relu6f
IdentityIdentityRelu6:activations:0*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*%
_input_shapes
:0:N J
&
_output_shapes
:0
 
_user_specified_nameinputs


O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1363

inputs-
)readvariableop_batch_normalization_4_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_4_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance0
,fusedbatchnormv3_batch_normalization_4_const
identityขAssignNewValueขAssignNewValue_1
ReadVariableOpReadVariableOp)readvariableop_batch_normalization_4_beta*
_output_shapes
:0*
dtype02
ReadVariableOpภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputs,fusedbatchnormv3_batch_normalization_4_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
ื#<2
FusedBatchNormV3ฑ
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_4_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_4/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueว
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_4/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1ฆ
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:+???????????????????????????0::::02 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs: 

_output_shapes
:0
๐	

4__inference_batch_normalization_1_layer_call_fn_5412

inputs
batch_normalization_1_beta%
!batch_normalization_1_moving_mean)
%batch_normalization_1_moving_variance
batch_normalization_1_const
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_1_beta!batch_normalization_1_moving_mean%batch_normalization_1_moving_variancebatch_normalization_1_const*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????0*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_10952
StatefulPartitionedCallจ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:+???????????????????????????0::::022
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs: 

_output_shapes
:0
?

P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_6475

inputs.
*readvariableop_batch_normalization_10_betaF
Bfusedbatchnormv3_readvariableop_batch_normalization_10_moving_meanL
Hfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance1
-fusedbatchnormv3_batch_normalization_10_const
identityขAssignNewValueขAssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_10_beta*
_output_shapes
:`*
dtype02
ReadVariableOpม
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_10_moving_mean*
_output_shapes
:`*
dtype02!
FusedBatchNormV3/ReadVariableOpห
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance*
_output_shapes
:`*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputs-fusedbatchnormv3_batch_normalization_10_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????`:`:`:`:`:*
epsilon%o:*
exponential_avg_factor%
ื#<2
FusedBatchNormV3ณ
AssignNewValueAssignVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_10_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*U
_classK
IGloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_10/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueษ
AssignNewValue_1AssignVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*[
_classQ
OMloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_10/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1ฆ
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+???????????????????????????`2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:+???????????????????????????`::::`2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+???????????????????????????`
 
_user_specified_nameinputs: 

_output_shapes
:`
	

4__inference_batch_normalization_4_layer_call_fn_5786

inputs
batch_normalization_4_beta%
!batch_normalization_4_moving_mean)
%batch_normalization_4_moving_variance
batch_normalization_4_const
identityขStatefulPartitionedCall๐
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_4_beta!batch_normalization_4_moving_mean%batch_normalization_4_moving_variancebatch_normalization_4_const*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_26212
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:0::::022
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:0
 
_user_specified_nameinputs: 

_output_shapes
:0
ฤ
u
'__inference_conv2d_4_layer_call_fn_5975

inputs
conv2d_4_kernel
identityขStatefulPartitionedCall์
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_4_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_27432
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*)
_input_shapes
:0:22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:0
 
_user_specified_nameinputs
โ
ใ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_6011

inputs.
*readvariableop_batch_normalization_6_gamma/
+readvariableop_1_batch_normalization_6_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_6_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_6_gamma*
_output_shapes
: *
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_6_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1ภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_6_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ม
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3g
IdentityIdentityFusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": :::::N J
&
_output_shapes
: 
 
_user_specified_nameinputs
ๅ

/__inference_depthwise_conv2d_layer_call_fn_1128

inputs%
!depthwise_conv2d_depthwise_kernel
identityขStatefulPartitionedCallก
StatefulPartitionedCallStatefulPartitionedCallinputs!depthwise_conv2d_depthwise_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_depthwise_conv2d_layer_call_and_return_conditional_losses_11242
StatefulPartitionedCallจ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+???????????????????????????0:22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs


O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_6279

inputs-
)readvariableop_batch_normalization_8_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_8_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance0
,fusedbatchnormv3_batch_normalization_8_const
identityขAssignNewValueขAssignNewValue_1
ReadVariableOpReadVariableOp)readvariableop_batch_normalization_8_beta*
_output_shapes
:0*
dtype02
ReadVariableOpภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputs,fusedbatchnormv3_batch_normalization_8_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
ื#<2
FusedBatchNormV3ฑ
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_8_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_8/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueว
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_8/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1ฆ
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:+???????????????????????????0::::02 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs: 

_output_shapes
:0
๔
_
&__inference_dropout_layer_call_fn_6841

inputs
identityขStatefulPartitionedCallฮ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_33272
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:@2

Identity"
identityIdentity:output:0*
_input_shapes

:@22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes

:@
 
_user_specified_nameinputs
ผ
q
%__inference_conv2d_layer_call_fn_5233

inputs
conv2d_kernel
identityขStatefulPartitionedCall่
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_22342
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*)
_input_shapes
:1(:22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:1(
 
_user_specified_nameinputs
ษ

B__inference_conv2d_7_layer_call_and_return_conditional_losses_3060

inputs)
%conv2d_readvariableop_conv2d_7_kernel
identity
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_7_kernel*&
_output_shapes
:@`*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:`*
paddingSAME*
strides
2
Conv2Db
IdentityIdentityConv2D:output:0*
T0*&
_output_shapes
:`2

Identity"
identityIdentity:output:0*)
_input_shapes
:@::N J
&
_output_shapes
:@
 
_user_specified_nameinputs
ง

O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2348

inputs-
)readvariableop_batch_normalization_1_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_1_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance0
,fusedbatchnormv3_batch_normalization_1_const
identityขAssignNewValueขAssignNewValue_1
ReadVariableOpReadVariableOp)readvariableop_batch_normalization_1_beta*
_output_shapes
:0*
dtype02
ReadVariableOpภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ใ
FusedBatchNormV3FusedBatchNormV3inputs,fusedbatchnormv3_batch_normalization_1_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
ื#<2
FusedBatchNormV3ฑ
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_1_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_1/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueว
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_1/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:0::::02 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:N J
&
_output_shapes
:0
 
_user_specified_nameinputs: 

_output_shapes
:0

ใ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5446

inputs-
)readvariableop_batch_normalization_1_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_1_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance0
,fusedbatchnormv3_batch_normalization_1_const
identity
ReadVariableOpReadVariableOp)readvariableop_batch_normalization_1_beta*
_output_shapes
:0*
dtype02
ReadVariableOpภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ี
FusedBatchNormV3FusedBatchNormV3inputs,fusedbatchnormv3_batch_normalization_1_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:0:0:0:0:0:*
epsilon%o:*
is_training( 2
FusedBatchNormV3g
IdentityIdentityFusedBatchNormV3:y:0*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:0::::0:N J
&
_output_shapes
:0
 
_user_specified_nameinputs: 

_output_shapes
:0
ํ

1__inference_depthwise_conv2d_1_layer_call_fn_1422

inputs'
#depthwise_conv2d_1_depthwise_kernel
identityขStatefulPartitionedCallฅ
StatefulPartitionedCallStatefulPartitionedCallinputs#depthwise_conv2d_1_depthwise_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_depthwise_conv2d_1_layer_call_and_return_conditional_losses_14182
StatefulPartitionedCallจ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+???????????????????????????0:22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs


O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5377

inputs-
)readvariableop_batch_normalization_1_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_1_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance0
,fusedbatchnormv3_batch_normalization_1_const
identityขAssignNewValueขAssignNewValue_1
ReadVariableOpReadVariableOp)readvariableop_batch_normalization_1_beta*
_output_shapes
:0*
dtype02
ReadVariableOpภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputs,fusedbatchnormv3_batch_normalization_1_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
ื#<2
FusedBatchNormV3ฑ
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_1_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_1/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueว
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_1/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1ฆ
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:+???????????????????????????0::::02 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs: 

_output_shapes
:0
ษ

B__inference_conv2d_6_layer_call_and_return_conditional_losses_6331

inputs)
%conv2d_readvariableop_conv2d_6_kernel
identity
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_6_kernel*&
_output_shapes
:0@*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingSAME*
strides
2
Conv2Db
IdentityIdentityConv2D:output:0*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*)
_input_shapes
:0::N J
&
_output_shapes
:0
 
_user_specified_nameinputs
ฯ
ใ
O__inference_batch_normalization_9_layer_call_and_return_conditional_losses_1889

inputs.
*readvariableop_batch_normalization_9_gamma/
+readvariableop_1_batch_normalization_9_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_9_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_9_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_9_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1ภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_9_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@:::::i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
๗
ใ
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1797

inputs-
)readvariableop_batch_normalization_8_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_8_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance0
,fusedbatchnormv3_batch_normalization_8_const
identity
ReadVariableOpReadVariableOp)readvariableop_batch_normalization_8_beta*
_output_shapes
:0*
dtype02
ReadVariableOpภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1๐
FusedBatchNormV3FusedBatchNormV3inputs,fusedbatchnormv3_batch_normalization_8_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:+???????????????????????????0::::0:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs: 

_output_shapes
:0
อ	

4__inference_batch_normalization_9_layer_call_fn_6391

inputs
batch_normalization_9_gamma
batch_normalization_9_beta%
!batch_normalization_9_moving_mean)
%batch_normalization_9_moving_variance
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_9_gammabatch_normalization_9_beta!batch_normalization_9_moving_mean%batch_normalization_9_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_9_layer_call_and_return_conditional_losses_18892
StatefulPartitionedCallจ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
ษ

B__inference_conv2d_2_layer_call_and_return_conditional_losses_2487

inputs)
%conv2d_readvariableop_conv2d_2_kernel
identity
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_2_kernel*&
_output_shapes
:0 *
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingSAME*
strides
2
Conv2Db
IdentityIdentityConv2D:output:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*)
_input_shapes
:0::N J
&
_output_shapes
:0
 
_user_specified_nameinputs
๗
ใ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_5934

inputs-
)readvariableop_batch_normalization_5_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_5_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance0
,fusedbatchnormv3_batch_normalization_5_const
identity
ReadVariableOpReadVariableOp)readvariableop_batch_normalization_5_beta*
_output_shapes
:0*
dtype02
ReadVariableOpภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_5_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1๐
FusedBatchNormV3FusedBatchNormV3inputs,fusedbatchnormv3_batch_normalization_5_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:+???????????????????????????0::::0:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs: 

_output_shapes
:0
๋
่
P__inference_batch_normalization_12_layer_call_and_return_conditional_losses_3270

inputs/
+readvariableop_batch_normalization_12_gamma0
,readvariableop_1_batch_normalization_12_betaF
Bfusedbatchnormv3_readvariableop_batch_normalization_12_moving_meanL
Hfusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance
identity
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_12_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_12_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1ม
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_12_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpห
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ม
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3g
IdentityIdentityFusedBatchNormV3:y:0*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*5
_input_shapes$
":@:::::N J
&
_output_shapes
:@
 
_user_specified_nameinputs
ื	

5__inference_batch_normalization_12_layer_call_fn_6807

inputs 
batch_normalization_12_gamma
batch_normalization_12_beta&
"batch_normalization_12_moving_mean*
&batch_normalization_12_moving_variance
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_12_gammabatch_normalization_12_beta"batch_normalization_12_moving_mean&batch_normalization_12_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_12_layer_call_and_return_conditional_losses_21832
StatefulPartitionedCallจ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs


P__inference_batch_normalization_12_layer_call_and_return_conditional_losses_3252

inputs/
+readvariableop_batch_normalization_12_gamma0
,readvariableop_1_batch_normalization_12_betaF
Bfusedbatchnormv3_readvariableop_batch_normalization_12_moving_meanL
Hfusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance
identityขAssignNewValueขAssignNewValue_1
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_12_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_12_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1ม
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_12_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpห
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ฯ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
ื#<2
FusedBatchNormV3ณ
AssignNewValueAssignVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_12_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*U
_classK
IGloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_12/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueษ
AssignNewValue_1AssignVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*[
_classQ
OMloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_12/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*5
_input_shapes$
":@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:N J
&
_output_shapes
:@
 
_user_specified_nameinputs

ใ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_6130

inputs-
)readvariableop_batch_normalization_7_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_7_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance0
,fusedbatchnormv3_batch_normalization_7_const
identity
ReadVariableOpReadVariableOp)readvariableop_batch_normalization_7_beta*
_output_shapes
:0*
dtype02
ReadVariableOpภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_7_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ี
FusedBatchNormV3FusedBatchNormV3inputs,fusedbatchnormv3_batch_normalization_7_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:0:0:0:0:0:*
epsilon%o:*
is_training( 2
FusedBatchNormV3g
IdentityIdentityFusedBatchNormV3:y:0*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:0::::0:N J
&
_output_shapes
:0
 
_user_specified_nameinputs: 

_output_shapes
:0

ใ1
F__inference_functional_1_layer_call_and_return_conditional_losses_4819

inputs.
*conv2d_conv2d_readvariableop_conv2d_kernel?
;batch_normalization_readvariableop_batch_normalization_betaW
Sbatch_normalization_fusedbatchnormv3_readvariableop_batch_normalization_moving_mean]
Ybatch_normalization_fusedbatchnormv3_readvariableop_1_batch_normalization_moving_varianceB
>batch_normalization_fusedbatchnormv3_batch_normalization_const2
.conv2d_1_conv2d_readvariableop_conv2d_1_kernelC
?batch_normalization_1_readvariableop_batch_normalization_1_beta[
Wbatch_normalization_1_fusedbatchnormv3_readvariableop_batch_normalization_1_moving_meana
]batch_normalization_1_fusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_varianceF
Bbatch_normalization_1_fusedbatchnormv3_batch_normalization_1_constO
Kdepthwise_conv2d_depthwise_readvariableop_depthwise_conv2d_depthwise_kernelC
?batch_normalization_2_readvariableop_batch_normalization_2_beta[
Wbatch_normalization_2_fusedbatchnormv3_readvariableop_batch_normalization_2_moving_meana
]batch_normalization_2_fusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_varianceF
Bbatch_normalization_2_fusedbatchnormv3_batch_normalization_2_const2
.conv2d_2_conv2d_readvariableop_conv2d_2_kernelD
@batch_normalization_3_readvariableop_batch_normalization_3_gammaE
Abatch_normalization_3_readvariableop_1_batch_normalization_3_beta[
Wbatch_normalization_3_fusedbatchnormv3_readvariableop_batch_normalization_3_moving_meana
]batch_normalization_3_fusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance2
.conv2d_3_conv2d_readvariableop_conv2d_3_kernelC
?batch_normalization_4_readvariableop_batch_normalization_4_beta[
Wbatch_normalization_4_fusedbatchnormv3_readvariableop_batch_normalization_4_moving_meana
]batch_normalization_4_fusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_varianceF
Bbatch_normalization_4_fusedbatchnormv3_batch_normalization_4_constS
Odepthwise_conv2d_1_depthwise_readvariableop_depthwise_conv2d_1_depthwise_kernelC
?batch_normalization_5_readvariableop_batch_normalization_5_beta[
Wbatch_normalization_5_fusedbatchnormv3_readvariableop_batch_normalization_5_moving_meana
]batch_normalization_5_fusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_varianceF
Bbatch_normalization_5_fusedbatchnormv3_batch_normalization_5_const2
.conv2d_4_conv2d_readvariableop_conv2d_4_kernelD
@batch_normalization_6_readvariableop_batch_normalization_6_gammaE
Abatch_normalization_6_readvariableop_1_batch_normalization_6_beta[
Wbatch_normalization_6_fusedbatchnormv3_readvariableop_batch_normalization_6_moving_meana
]batch_normalization_6_fusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance2
.conv2d_5_conv2d_readvariableop_conv2d_5_kernelC
?batch_normalization_7_readvariableop_batch_normalization_7_beta[
Wbatch_normalization_7_fusedbatchnormv3_readvariableop_batch_normalization_7_moving_meana
]batch_normalization_7_fusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_varianceF
Bbatch_normalization_7_fusedbatchnormv3_batch_normalization_7_constS
Odepthwise_conv2d_2_depthwise_readvariableop_depthwise_conv2d_2_depthwise_kernelC
?batch_normalization_8_readvariableop_batch_normalization_8_beta[
Wbatch_normalization_8_fusedbatchnormv3_readvariableop_batch_normalization_8_moving_meana
]batch_normalization_8_fusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_varianceF
Bbatch_normalization_8_fusedbatchnormv3_batch_normalization_8_const2
.conv2d_6_conv2d_readvariableop_conv2d_6_kernelD
@batch_normalization_9_readvariableop_batch_normalization_9_gammaE
Abatch_normalization_9_readvariableop_1_batch_normalization_9_beta[
Wbatch_normalization_9_fusedbatchnormv3_readvariableop_batch_normalization_9_moving_meana
]batch_normalization_9_fusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance2
.conv2d_7_conv2d_readvariableop_conv2d_7_kernelE
Abatch_normalization_10_readvariableop_batch_normalization_10_beta]
Ybatch_normalization_10_fusedbatchnormv3_readvariableop_batch_normalization_10_moving_meanc
_batch_normalization_10_fusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_varianceH
Dbatch_normalization_10_fusedbatchnormv3_batch_normalization_10_constS
Odepthwise_conv2d_3_depthwise_readvariableop_depthwise_conv2d_3_depthwise_kernelE
Abatch_normalization_11_readvariableop_batch_normalization_11_beta]
Ybatch_normalization_11_fusedbatchnormv3_readvariableop_batch_normalization_11_moving_meanc
_batch_normalization_11_fusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_varianceH
Dbatch_normalization_11_fusedbatchnormv3_batch_normalization_11_const2
.conv2d_8_conv2d_readvariableop_conv2d_8_kernelF
Bbatch_normalization_12_readvariableop_batch_normalization_12_gammaG
Cbatch_normalization_12_readvariableop_1_batch_normalization_12_beta]
Ybatch_normalization_12_fusedbatchnormv3_readvariableop_batch_normalization_12_moving_meanc
_batch_normalization_12_fusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance,
(dense_matmul_readvariableop_dense_kernel+
'dense_biasadd_readvariableop_dense_bias
identityข"batch_normalization/AssignNewValueข$batch_normalization/AssignNewValue_1ข$batch_normalization_1/AssignNewValueข&batch_normalization_1/AssignNewValue_1ข%batch_normalization_10/AssignNewValueข'batch_normalization_10/AssignNewValue_1ข%batch_normalization_11/AssignNewValueข'batch_normalization_11/AssignNewValue_1ข%batch_normalization_12/AssignNewValueข'batch_normalization_12/AssignNewValue_1ข$batch_normalization_2/AssignNewValueข&batch_normalization_2/AssignNewValue_1ข$batch_normalization_3/AssignNewValueข&batch_normalization_3/AssignNewValue_1ข$batch_normalization_4/AssignNewValueข&batch_normalization_4/AssignNewValue_1ข$batch_normalization_5/AssignNewValueข&batch_normalization_5/AssignNewValue_1ข$batch_normalization_6/AssignNewValueข&batch_normalization_6/AssignNewValue_1ข$batch_normalization_7/AssignNewValueข&batch_normalization_7/AssignNewValue_1ข$batch_normalization_8/AssignNewValueข&batch_normalization_8/AssignNewValue_1ข$batch_normalization_9/AssignNewValueข&batch_normalization_9/AssignNewValue_1
%tf_op_layer_ExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%tf_op_layer_ExpandDims/ExpandDims/dimฬ
!tf_op_layer_ExpandDims/ExpandDims
ExpandDimsinputs.tf_op_layer_ExpandDims/ExpandDims/dim:output:0*
T0*
_cloned(*&
_output_shapes
:1(2#
!tf_op_layer_ExpandDims/ExpandDimsฏ
conv2d/Conv2D/ReadVariableOpReadVariableOp*conv2d_conv2d_readvariableop_conv2d_kernel*&
_output_shapes
:( *
dtype02
conv2d/Conv2D/ReadVariableOpิ
conv2d/Conv2DConv2D*tf_op_layer_ExpandDims/ExpandDims:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
2
conv2d/Conv2Dภ
"batch_normalization/ReadVariableOpReadVariableOp;batch_normalization_readvariableop_batch_normalization_beta*
_output_shapes
: *
dtype02$
"batch_normalization/ReadVariableOp๚
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpSbatch_normalization_fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
_output_shapes
: *
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpYbatch_normalization_fusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
_output_shapes
: *
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1้
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/Conv2D:output:0>batch_normalization_fusedbatchnormv3_batch_normalization_const*batch_normalization/ReadVariableOp:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
exponential_avg_factor%
ื#<2&
$batch_normalization/FusedBatchNormV3ฅ
"batch_normalization/AssignNewValueAssignVariableOpSbatch_normalization_fusedbatchnormv3_readvariableop_batch_normalization_moving_mean1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*f
_class\
ZXloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp/batch_normalization/moving_mean*
_output_shapes
 *
dtype02$
"batch_normalization/AssignNewValueป
$batch_normalization/AssignNewValue_1AssignVariableOpYbatch_normalization_fusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*l
_classb
`^loc:@batch_normalization/FusedBatchNormV3/ReadVariableOp_1/batch_normalization/moving_variance*
_output_shapes
 *
dtype02&
$batch_normalization/AssignNewValue_1~
re_lu/Relu6Relu6(batch_normalization/FusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2
re_lu/Relu6ท
conv2d_1/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_conv2d_readvariableop_conv2d_1_kernel*&
_output_shapes
: 0*
dtype02 
conv2d_1/Conv2D/ReadVariableOpศ
conv2d_1/Conv2DConv2Dre_lu/Relu6:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:0*
paddingSAME*
strides
2
conv2d_1/Conv2Dศ
$batch_normalization_1/ReadVariableOpReadVariableOp?batch_normalization_1_readvariableop_batch_normalization_1_beta*
_output_shapes
:0*
dtype02&
$batch_normalization_1/ReadVariableOp
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_1_fusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
:0*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_1_fusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
_output_shapes
:0*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1๙
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/Conv2D:output:0Bbatch_normalization_1_fusedbatchnormv3_batch_normalization_1_const,batch_normalization_1/ReadVariableOp:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
ื#<2(
&batch_normalization_1/FusedBatchNormV3ต
$batch_normalization_1/AssignNewValueAssignVariableOpWbatch_normalization_1_fusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*j
_class`
^\loc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp/batch_normalization_1/moving_mean*
_output_shapes
 *
dtype02&
$batch_normalization_1/AssignNewValueห
&batch_normalization_1/AssignNewValue_1AssignVariableOp]batch_normalization_1_fusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*p
_classf
dbloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_1/moving_variance*
_output_shapes
 *
dtype02(
&batch_normalization_1/AssignNewValue_1
re_lu_1/Relu6Relu6*batch_normalization_1/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:02
re_lu_1/Relu6๊
)depthwise_conv2d/depthwise/ReadVariableOpReadVariableOpKdepthwise_conv2d_depthwise_readvariableop_depthwise_conv2d_depthwise_kernel*&
_output_shapes
:0*
dtype02+
)depthwise_conv2d/depthwise/ReadVariableOp
 depthwise_conv2d/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      0      2"
 depthwise_conv2d/depthwise/Shapeฅ
(depthwise_conv2d/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2*
(depthwise_conv2d/depthwise/dilation_rate๚
depthwise_conv2d/depthwiseDepthwiseConv2dNativere_lu_1/Relu6:activations:01depthwise_conv2d/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:0*
paddingSAME*
strides
2
depthwise_conv2d/depthwiseศ
$batch_normalization_2/ReadVariableOpReadVariableOp?batch_normalization_2_readvariableop_batch_normalization_2_beta*
_output_shapes
:0*
dtype02&
$batch_normalization_2/ReadVariableOp
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_2_fusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
:0*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_2_fusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
_output_shapes
:0*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3#depthwise_conv2d/depthwise:output:0Bbatch_normalization_2_fusedbatchnormv3_batch_normalization_2_const,batch_normalization_2/ReadVariableOp:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
ื#<2(
&batch_normalization_2/FusedBatchNormV3ต
$batch_normalization_2/AssignNewValueAssignVariableOpWbatch_normalization_2_fusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*j
_class`
^\loc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp/batch_normalization_2/moving_mean*
_output_shapes
 *
dtype02&
$batch_normalization_2/AssignNewValueห
&batch_normalization_2/AssignNewValue_1AssignVariableOp]batch_normalization_2_fusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*p
_classf
dbloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_2/moving_variance*
_output_shapes
 *
dtype02(
&batch_normalization_2/AssignNewValue_1
re_lu_2/Relu6Relu6*batch_normalization_2/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:02
re_lu_2/Relu6ท
conv2d_2/Conv2D/ReadVariableOpReadVariableOp.conv2d_2_conv2d_readvariableop_conv2d_2_kernel*&
_output_shapes
:0 *
dtype02 
conv2d_2/Conv2D/ReadVariableOpส
conv2d_2/Conv2DConv2Dre_lu_2/Relu6:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingSAME*
strides
2
conv2d_2/Conv2Dษ
$batch_normalization_3/ReadVariableOpReadVariableOp@batch_normalization_3_readvariableop_batch_normalization_3_gamma*
_output_shapes
: *
dtype02&
$batch_normalization_3/ReadVariableOpฮ
&batch_normalization_3/ReadVariableOp_1ReadVariableOpAbatch_normalization_3_readvariableop_1_batch_normalization_3_beta*
_output_shapes
: *
dtype02(
&batch_normalization_3/ReadVariableOp_1
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_3_fusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
_output_shapes
: *
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_3_fusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
_output_shapes
: *
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ๅ
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_2/Conv2D:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
exponential_avg_factor%
ื#<2(
&batch_normalization_3/FusedBatchNormV3ต
$batch_normalization_3/AssignNewValueAssignVariableOpWbatch_normalization_3_fusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*j
_class`
^\loc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp/batch_normalization_3/moving_mean*
_output_shapes
 *
dtype02&
$batch_normalization_3/AssignNewValueห
&batch_normalization_3/AssignNewValue_1AssignVariableOp]batch_normalization_3_fusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance7batch_normalization_3/FusedBatchNormV3:batch_variance:08^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*p
_classf
dbloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_3/moving_variance*
_output_shapes
 *
dtype02(
&batch_normalization_3/AssignNewValue_1
add/addAddV2re_lu/Relu6:activations:0*batch_normalization_3/FusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2	
add/addท
conv2d_3/Conv2D/ReadVariableOpReadVariableOp.conv2d_3_conv2d_readvariableop_conv2d_3_kernel*&
_output_shapes
: 0*
dtype02 
conv2d_3/Conv2D/ReadVariableOpบ
conv2d_3/Conv2DConv2Dadd/add:z:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:0*
paddingSAME*
strides
2
conv2d_3/Conv2Dศ
$batch_normalization_4/ReadVariableOpReadVariableOp?batch_normalization_4_readvariableop_batch_normalization_4_beta*
_output_shapes
:0*
dtype02&
$batch_normalization_4/ReadVariableOp
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_4_fusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
_output_shapes
:0*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_4_fusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
_output_shapes
:0*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1๙
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_3/Conv2D:output:0Bbatch_normalization_4_fusedbatchnormv3_batch_normalization_4_const,batch_normalization_4/ReadVariableOp:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
ื#<2(
&batch_normalization_4/FusedBatchNormV3ต
$batch_normalization_4/AssignNewValueAssignVariableOpWbatch_normalization_4_fusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean3batch_normalization_4/FusedBatchNormV3:batch_mean:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp*j
_class`
^\loc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp/batch_normalization_4/moving_mean*
_output_shapes
 *
dtype02&
$batch_normalization_4/AssignNewValueห
&batch_normalization_4/AssignNewValue_1AssignVariableOp]batch_normalization_4_fusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance7batch_normalization_4/FusedBatchNormV3:batch_variance:08^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*p
_classf
dbloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_4/moving_variance*
_output_shapes
 *
dtype02(
&batch_normalization_4/AssignNewValue_1
re_lu_3/Relu6Relu6*batch_normalization_4/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:02
re_lu_3/Relu6๒
+depthwise_conv2d_1/depthwise/ReadVariableOpReadVariableOpOdepthwise_conv2d_1_depthwise_readvariableop_depthwise_conv2d_1_depthwise_kernel*&
_output_shapes
:0*
dtype02-
+depthwise_conv2d_1/depthwise/ReadVariableOpก
"depthwise_conv2d_1/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      0      2$
"depthwise_conv2d_1/depthwise/Shapeฉ
*depthwise_conv2d_1/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2,
*depthwise_conv2d_1/depthwise/dilation_rate
depthwise_conv2d_1/depthwiseDepthwiseConv2dNativere_lu_3/Relu6:activations:03depthwise_conv2d_1/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:0*
paddingSAME*
strides
2
depthwise_conv2d_1/depthwiseศ
$batch_normalization_5/ReadVariableOpReadVariableOp?batch_normalization_5_readvariableop_batch_normalization_5_beta*
_output_shapes
:0*
dtype02&
$batch_normalization_5/ReadVariableOp
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_5_fusedbatchnormv3_readvariableop_batch_normalization_5_moving_mean*
_output_shapes
:0*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_5_fusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance*
_output_shapes
:0*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3%depthwise_conv2d_1/depthwise:output:0Bbatch_normalization_5_fusedbatchnormv3_batch_normalization_5_const,batch_normalization_5/ReadVariableOp:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
ื#<2(
&batch_normalization_5/FusedBatchNormV3ต
$batch_normalization_5/AssignNewValueAssignVariableOpWbatch_normalization_5_fusedbatchnormv3_readvariableop_batch_normalization_5_moving_mean3batch_normalization_5/FusedBatchNormV3:batch_mean:06^batch_normalization_5/FusedBatchNormV3/ReadVariableOp*j
_class`
^\loc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp/batch_normalization_5/moving_mean*
_output_shapes
 *
dtype02&
$batch_normalization_5/AssignNewValueห
&batch_normalization_5/AssignNewValue_1AssignVariableOp]batch_normalization_5_fusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance7batch_normalization_5/FusedBatchNormV3:batch_variance:08^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*p
_classf
dbloc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_5/moving_variance*
_output_shapes
 *
dtype02(
&batch_normalization_5/AssignNewValue_1
re_lu_4/Relu6Relu6*batch_normalization_5/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:02
re_lu_4/Relu6ท
conv2d_4/Conv2D/ReadVariableOpReadVariableOp.conv2d_4_conv2d_readvariableop_conv2d_4_kernel*&
_output_shapes
:0 *
dtype02 
conv2d_4/Conv2D/ReadVariableOpส
conv2d_4/Conv2DConv2Dre_lu_4/Relu6:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingSAME*
strides
2
conv2d_4/Conv2Dษ
$batch_normalization_6/ReadVariableOpReadVariableOp@batch_normalization_6_readvariableop_batch_normalization_6_gamma*
_output_shapes
: *
dtype02&
$batch_normalization_6/ReadVariableOpฮ
&batch_normalization_6/ReadVariableOp_1ReadVariableOpAbatch_normalization_6_readvariableop_1_batch_normalization_6_beta*
_output_shapes
: *
dtype02(
&batch_normalization_6/ReadVariableOp_1
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_6_fusedbatchnormv3_readvariableop_batch_normalization_6_moving_mean*
_output_shapes
: *
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_6_fusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance*
_output_shapes
: *
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ๅ
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv2d_4/Conv2D:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
exponential_avg_factor%
ื#<2(
&batch_normalization_6/FusedBatchNormV3ต
$batch_normalization_6/AssignNewValueAssignVariableOpWbatch_normalization_6_fusedbatchnormv3_readvariableop_batch_normalization_6_moving_mean3batch_normalization_6/FusedBatchNormV3:batch_mean:06^batch_normalization_6/FusedBatchNormV3/ReadVariableOp*j
_class`
^\loc:@batch_normalization_6/FusedBatchNormV3/ReadVariableOp/batch_normalization_6/moving_mean*
_output_shapes
 *
dtype02&
$batch_normalization_6/AssignNewValueห
&batch_normalization_6/AssignNewValue_1AssignVariableOp]batch_normalization_6_fusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance7batch_normalization_6/FusedBatchNormV3:batch_variance:08^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1*p
_classf
dbloc:@batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_6/moving_variance*
_output_shapes
 *
dtype02(
&batch_normalization_6/AssignNewValue_1ท
conv2d_5/Conv2D/ReadVariableOpReadVariableOp.conv2d_5_conv2d_readvariableop_conv2d_5_kernel*&
_output_shapes
: 0*
dtype02 
conv2d_5/Conv2D/ReadVariableOpู
conv2d_5/Conv2DConv2D*batch_normalization_6/FusedBatchNormV3:y:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:0*
paddingSAME*
strides
2
conv2d_5/Conv2Dศ
$batch_normalization_7/ReadVariableOpReadVariableOp?batch_normalization_7_readvariableop_batch_normalization_7_beta*
_output_shapes
:0*
dtype02&
$batch_normalization_7/ReadVariableOp
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_7_fusedbatchnormv3_readvariableop_batch_normalization_7_moving_mean*
_output_shapes
:0*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_7_fusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance*
_output_shapes
:0*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1๙
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3conv2d_5/Conv2D:output:0Bbatch_normalization_7_fusedbatchnormv3_batch_normalization_7_const,batch_normalization_7/ReadVariableOp:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
ื#<2(
&batch_normalization_7/FusedBatchNormV3ต
$batch_normalization_7/AssignNewValueAssignVariableOpWbatch_normalization_7_fusedbatchnormv3_readvariableop_batch_normalization_7_moving_mean3batch_normalization_7/FusedBatchNormV3:batch_mean:06^batch_normalization_7/FusedBatchNormV3/ReadVariableOp*j
_class`
^\loc:@batch_normalization_7/FusedBatchNormV3/ReadVariableOp/batch_normalization_7/moving_mean*
_output_shapes
 *
dtype02&
$batch_normalization_7/AssignNewValueห
&batch_normalization_7/AssignNewValue_1AssignVariableOp]batch_normalization_7_fusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance7batch_normalization_7/FusedBatchNormV3:batch_variance:08^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1*p
_classf
dbloc:@batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_7/moving_variance*
_output_shapes
 *
dtype02(
&batch_normalization_7/AssignNewValue_1
re_lu_5/Relu6Relu6*batch_normalization_7/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:02
re_lu_5/Relu6๒
+depthwise_conv2d_2/depthwise/ReadVariableOpReadVariableOpOdepthwise_conv2d_2_depthwise_readvariableop_depthwise_conv2d_2_depthwise_kernel*&
_output_shapes
:0*
dtype02-
+depthwise_conv2d_2/depthwise/ReadVariableOpก
"depthwise_conv2d_2/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      0      2$
"depthwise_conv2d_2/depthwise/Shapeฉ
*depthwise_conv2d_2/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2,
*depthwise_conv2d_2/depthwise/dilation_rate
depthwise_conv2d_2/depthwiseDepthwiseConv2dNativere_lu_5/Relu6:activations:03depthwise_conv2d_2/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:0*
paddingSAME*
strides
2
depthwise_conv2d_2/depthwiseศ
$batch_normalization_8/ReadVariableOpReadVariableOp?batch_normalization_8_readvariableop_batch_normalization_8_beta*
_output_shapes
:0*
dtype02&
$batch_normalization_8/ReadVariableOp
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_8_fusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean*
_output_shapes
:0*
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_8_fusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance*
_output_shapes
:0*
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3%depthwise_conv2d_2/depthwise:output:0Bbatch_normalization_8_fusedbatchnormv3_batch_normalization_8_const,batch_normalization_8/ReadVariableOp:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
ื#<2(
&batch_normalization_8/FusedBatchNormV3ต
$batch_normalization_8/AssignNewValueAssignVariableOpWbatch_normalization_8_fusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean3batch_normalization_8/FusedBatchNormV3:batch_mean:06^batch_normalization_8/FusedBatchNormV3/ReadVariableOp*j
_class`
^\loc:@batch_normalization_8/FusedBatchNormV3/ReadVariableOp/batch_normalization_8/moving_mean*
_output_shapes
 *
dtype02&
$batch_normalization_8/AssignNewValueห
&batch_normalization_8/AssignNewValue_1AssignVariableOp]batch_normalization_8_fusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance7batch_normalization_8/FusedBatchNormV3:batch_variance:08^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1*p
_classf
dbloc:@batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_8/moving_variance*
_output_shapes
 *
dtype02(
&batch_normalization_8/AssignNewValue_1
re_lu_6/Relu6Relu6*batch_normalization_8/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:02
re_lu_6/Relu6ท
conv2d_6/Conv2D/ReadVariableOpReadVariableOp.conv2d_6_conv2d_readvariableop_conv2d_6_kernel*&
_output_shapes
:0@*
dtype02 
conv2d_6/Conv2D/ReadVariableOpส
conv2d_6/Conv2DConv2Dre_lu_6/Relu6:activations:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingSAME*
strides
2
conv2d_6/Conv2Dษ
$batch_normalization_9/ReadVariableOpReadVariableOp@batch_normalization_9_readvariableop_batch_normalization_9_gamma*
_output_shapes
:@*
dtype02&
$batch_normalization_9/ReadVariableOpฮ
&batch_normalization_9/ReadVariableOp_1ReadVariableOpAbatch_normalization_9_readvariableop_1_batch_normalization_9_beta*
_output_shapes
:@*
dtype02(
&batch_normalization_9/ReadVariableOp_1
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_9_fusedbatchnormv3_readvariableop_batch_normalization_9_moving_mean*
_output_shapes
:@*
dtype027
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_9_fusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance*
_output_shapes
:@*
dtype029
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ๅ
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3conv2d_6/Conv2D:output:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
ื#<2(
&batch_normalization_9/FusedBatchNormV3ต
$batch_normalization_9/AssignNewValueAssignVariableOpWbatch_normalization_9_fusedbatchnormv3_readvariableop_batch_normalization_9_moving_mean3batch_normalization_9/FusedBatchNormV3:batch_mean:06^batch_normalization_9/FusedBatchNormV3/ReadVariableOp*j
_class`
^\loc:@batch_normalization_9/FusedBatchNormV3/ReadVariableOp/batch_normalization_9/moving_mean*
_output_shapes
 *
dtype02&
$batch_normalization_9/AssignNewValueห
&batch_normalization_9/AssignNewValue_1AssignVariableOp]batch_normalization_9_fusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance7batch_normalization_9/FusedBatchNormV3:batch_variance:08^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1*p
_classf
dbloc:@batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_9/moving_variance*
_output_shapes
 *
dtype02(
&batch_normalization_9/AssignNewValue_1ท
conv2d_7/Conv2D/ReadVariableOpReadVariableOp.conv2d_7_conv2d_readvariableop_conv2d_7_kernel*&
_output_shapes
:@`*
dtype02 
conv2d_7/Conv2D/ReadVariableOpู
conv2d_7/Conv2DConv2D*batch_normalization_9/FusedBatchNormV3:y:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:`*
paddingSAME*
strides
2
conv2d_7/Conv2Dฬ
%batch_normalization_10/ReadVariableOpReadVariableOpAbatch_normalization_10_readvariableop_batch_normalization_10_beta*
_output_shapes
:`*
dtype02'
%batch_normalization_10/ReadVariableOp
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOpYbatch_normalization_10_fusedbatchnormv3_readvariableop_batch_normalization_10_moving_mean*
_output_shapes
:`*
dtype028
6batch_normalization_10/FusedBatchNormV3/ReadVariableOp
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp_batch_normalization_10_fusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance*
_output_shapes
:`*
dtype02:
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1
'batch_normalization_10/FusedBatchNormV3FusedBatchNormV3conv2d_7/Conv2D:output:0Dbatch_normalization_10_fusedbatchnormv3_batch_normalization_10_const-batch_normalization_10/ReadVariableOp:value:0>batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:`:`:`:`:`:*
epsilon%o:*
exponential_avg_factor%
ื#<2)
'batch_normalization_10/FusedBatchNormV3ฝ
%batch_normalization_10/AssignNewValueAssignVariableOpYbatch_normalization_10_fusedbatchnormv3_readvariableop_batch_normalization_10_moving_mean4batch_normalization_10/FusedBatchNormV3:batch_mean:07^batch_normalization_10/FusedBatchNormV3/ReadVariableOp*l
_classb
`^loc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp/batch_normalization_10/moving_mean*
_output_shapes
 *
dtype02'
%batch_normalization_10/AssignNewValueำ
'batch_normalization_10/AssignNewValue_1AssignVariableOp_batch_normalization_10_fusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance8batch_normalization_10/FusedBatchNormV3:batch_variance:09^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1*r
_classh
fdloc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_10/moving_variance*
_output_shapes
 *
dtype02)
'batch_normalization_10/AssignNewValue_1
re_lu_7/Relu6Relu6+batch_normalization_10/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:`2
re_lu_7/Relu6๒
+depthwise_conv2d_3/depthwise/ReadVariableOpReadVariableOpOdepthwise_conv2d_3_depthwise_readvariableop_depthwise_conv2d_3_depthwise_kernel*&
_output_shapes
:`*
dtype02-
+depthwise_conv2d_3/depthwise/ReadVariableOpก
"depthwise_conv2d_3/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      `      2$
"depthwise_conv2d_3/depthwise/Shapeฉ
*depthwise_conv2d_3/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2,
*depthwise_conv2d_3/depthwise/dilation_rate
depthwise_conv2d_3/depthwiseDepthwiseConv2dNativere_lu_7/Relu6:activations:03depthwise_conv2d_3/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:`*
paddingSAME*
strides
2
depthwise_conv2d_3/depthwiseฬ
%batch_normalization_11/ReadVariableOpReadVariableOpAbatch_normalization_11_readvariableop_batch_normalization_11_beta*
_output_shapes
:`*
dtype02'
%batch_normalization_11/ReadVariableOp
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOpYbatch_normalization_11_fusedbatchnormv3_readvariableop_batch_normalization_11_moving_mean*
_output_shapes
:`*
dtype028
6batch_normalization_11/FusedBatchNormV3/ReadVariableOp
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp_batch_normalization_11_fusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance*
_output_shapes
:`*
dtype02:
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1
'batch_normalization_11/FusedBatchNormV3FusedBatchNormV3%depthwise_conv2d_3/depthwise:output:0Dbatch_normalization_11_fusedbatchnormv3_batch_normalization_11_const-batch_normalization_11/ReadVariableOp:value:0>batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:`:`:`:`:`:*
epsilon%o:*
exponential_avg_factor%
ื#<2)
'batch_normalization_11/FusedBatchNormV3ฝ
%batch_normalization_11/AssignNewValueAssignVariableOpYbatch_normalization_11_fusedbatchnormv3_readvariableop_batch_normalization_11_moving_mean4batch_normalization_11/FusedBatchNormV3:batch_mean:07^batch_normalization_11/FusedBatchNormV3/ReadVariableOp*l
_classb
`^loc:@batch_normalization_11/FusedBatchNormV3/ReadVariableOp/batch_normalization_11/moving_mean*
_output_shapes
 *
dtype02'
%batch_normalization_11/AssignNewValueำ
'batch_normalization_11/AssignNewValue_1AssignVariableOp_batch_normalization_11_fusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance8batch_normalization_11/FusedBatchNormV3:batch_variance:09^batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1*r
_classh
fdloc:@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_11/moving_variance*
_output_shapes
 *
dtype02)
'batch_normalization_11/AssignNewValue_1
re_lu_8/Relu6Relu6+batch_normalization_11/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:`2
re_lu_8/Relu6ท
conv2d_8/Conv2D/ReadVariableOpReadVariableOp.conv2d_8_conv2d_readvariableop_conv2d_8_kernel*&
_output_shapes
:`@*
dtype02 
conv2d_8/Conv2D/ReadVariableOpส
conv2d_8/Conv2DConv2Dre_lu_8/Relu6:activations:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingSAME*
strides
2
conv2d_8/Conv2Dอ
%batch_normalization_12/ReadVariableOpReadVariableOpBbatch_normalization_12_readvariableop_batch_normalization_12_gamma*
_output_shapes
:@*
dtype02'
%batch_normalization_12/ReadVariableOpา
'batch_normalization_12/ReadVariableOp_1ReadVariableOpCbatch_normalization_12_readvariableop_1_batch_normalization_12_beta*
_output_shapes
:@*
dtype02)
'batch_normalization_12/ReadVariableOp_1
6batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOpYbatch_normalization_12_fusedbatchnormv3_readvariableop_batch_normalization_12_moving_mean*
_output_shapes
:@*
dtype028
6batch_normalization_12/FusedBatchNormV3/ReadVariableOp
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp_batch_normalization_12_fusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance*
_output_shapes
:@*
dtype02:
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1๋
'batch_normalization_12/FusedBatchNormV3FusedBatchNormV3conv2d_8/Conv2D:output:0-batch_normalization_12/ReadVariableOp:value:0/batch_normalization_12/ReadVariableOp_1:value:0>batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
ื#<2)
'batch_normalization_12/FusedBatchNormV3ฝ
%batch_normalization_12/AssignNewValueAssignVariableOpYbatch_normalization_12_fusedbatchnormv3_readvariableop_batch_normalization_12_moving_mean4batch_normalization_12/FusedBatchNormV3:batch_mean:07^batch_normalization_12/FusedBatchNormV3/ReadVariableOp*l
_classb
`^loc:@batch_normalization_12/FusedBatchNormV3/ReadVariableOp/batch_normalization_12/moving_mean*
_output_shapes
 *
dtype02'
%batch_normalization_12/AssignNewValueำ
'batch_normalization_12/AssignNewValue_1AssignVariableOp_batch_normalization_12_fusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance8batch_normalization_12/FusedBatchNormV3:batch_variance:09^batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1*r
_classh
fdloc:@batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_12/moving_variance*
_output_shapes
 *
dtype02)
'batch_normalization_12/AssignNewValue_1ฉ
	add_1/addAddV2*batch_normalization_9/FusedBatchNormV3:y:0+batch_normalization_12/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:@2
	add_1/addณ
/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      21
/global_average_pooling2d/Mean/reduction_indicesธ
global_average_pooling2d/MeanMeanadd_1/add:z:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*
_output_shapes

:@2
global_average_pooling2d/Means
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/dropout/Constข
dropout/dropout/MulMul&global_average_pooling2d/Mean:output:0dropout/dropout/Const:output:0*
T0*
_output_shapes

:@2
dropout/dropout/Mul
dropout/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2
dropout/dropout/Shapeร
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*
_output_shapes

:@*
dtype02.
,dropout/dropout/random_uniform/RandomUniform
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *อฬL>2 
dropout/dropout/GreaterEqual/yี
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes

:@2
dropout/dropout/GreaterEqual
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:@2
dropout/dropout/Cast
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*
_output_shapes

:@2
dropout/dropout/Mul_1ฃ
dense/MatMul/ReadVariableOpReadVariableOp(dense_matmul_readvariableop_dense_kernel*
_output_shapes

:@*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMuldropout/dropout/Mul_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp'dense_biasadd_readvariableop_dense_bias*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense/BiasAdd๓
IdentityIdentitydense/BiasAdd:output:0#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_1&^batch_normalization_10/AssignNewValue(^batch_normalization_10/AssignNewValue_1&^batch_normalization_11/AssignNewValue(^batch_normalization_11/AssignNewValue_1&^batch_normalization_12/AssignNewValue(^batch_normalization_12/AssignNewValue_1%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_1%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_1%^batch_normalization_4/AssignNewValue'^batch_normalization_4/AssignNewValue_1%^batch_normalization_5/AssignNewValue'^batch_normalization_5/AssignNewValue_1%^batch_normalization_6/AssignNewValue'^batch_normalization_6/AssignNewValue_1%^batch_normalization_7/AssignNewValue'^batch_normalization_7/AssignNewValue_1%^batch_normalization_8/AssignNewValue'^batch_normalization_8/AssignNewValue_1%^batch_normalization_9/AssignNewValue'^batch_normalization_9/AssignNewValue_1*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*ม
_input_shapesฏ
ฌ:1(::::: :::::0:::::0::::::::::0:::::0::::::::::0:::::0::::::::::`:::::`:::::::2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12N
%batch_normalization_10/AssignNewValue%batch_normalization_10/AssignNewValue2R
'batch_normalization_10/AssignNewValue_1'batch_normalization_10/AssignNewValue_12N
%batch_normalization_11/AssignNewValue%batch_normalization_11/AssignNewValue2R
'batch_normalization_11/AssignNewValue_1'batch_normalization_11/AssignNewValue_12N
%batch_normalization_12/AssignNewValue%batch_normalization_12/AssignNewValue2R
'batch_normalization_12/AssignNewValue_1'batch_normalization_12/AssignNewValue_12L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_12L
$batch_normalization_3/AssignNewValue$batch_normalization_3/AssignNewValue2P
&batch_normalization_3/AssignNewValue_1&batch_normalization_3/AssignNewValue_12L
$batch_normalization_4/AssignNewValue$batch_normalization_4/AssignNewValue2P
&batch_normalization_4/AssignNewValue_1&batch_normalization_4/AssignNewValue_12L
$batch_normalization_5/AssignNewValue$batch_normalization_5/AssignNewValue2P
&batch_normalization_5/AssignNewValue_1&batch_normalization_5/AssignNewValue_12L
$batch_normalization_6/AssignNewValue$batch_normalization_6/AssignNewValue2P
&batch_normalization_6/AssignNewValue_1&batch_normalization_6/AssignNewValue_12L
$batch_normalization_7/AssignNewValue$batch_normalization_7/AssignNewValue2P
&batch_normalization_7/AssignNewValue_1&batch_normalization_7/AssignNewValue_12L
$batch_normalization_8/AssignNewValue$batch_normalization_8/AssignNewValue2P
&batch_normalization_8/AssignNewValue_1&batch_normalization_8/AssignNewValue_12L
$batch_normalization_9/AssignNewValue$batch_normalization_9/AssignNewValue2P
&batch_normalization_9/AssignNewValue_1&batch_normalization_9/AssignNewValue_1:S O
+
_output_shapes
:?????????1(
 
_user_specified_nameinputs: 

_output_shapes
: : 


_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0: (

_output_shapes
:0: -

_output_shapes
:0: 7

_output_shapes
:`: <

_output_shapes
:`
	

5__inference_batch_normalization_11_layer_call_fn_6615

inputs
batch_normalization_11_beta&
"batch_normalization_11_moving_mean*
&batch_normalization_11_moving_variance 
batch_normalization_11_const
identityขStatefulPartitionedCall๓
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_11_beta"batch_normalization_11_moving_mean&batch_normalization_11_moving_variancebatch_normalization_11_const*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:`*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_31642
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:`2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:`::::`22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:`
 
_user_specified_nameinputs: 

_output_shapes
:`
ฐๆ
?%
F__inference_functional_1_layer_call_and_return_conditional_losses_3779

inputs
conv2d_conv2d_kernel0
,batch_normalization_batch_normalization_beta7
3batch_normalization_batch_normalization_moving_mean;
7batch_normalization_batch_normalization_moving_variance1
-batch_normalization_batch_normalization_const
conv2d_1_conv2d_1_kernel4
0batch_normalization_1_batch_normalization_1_beta;
7batch_normalization_1_batch_normalization_1_moving_mean?
;batch_normalization_1_batch_normalization_1_moving_variance5
1batch_normalization_1_batch_normalization_1_const6
2depthwise_conv2d_depthwise_conv2d_depthwise_kernel4
0batch_normalization_2_batch_normalization_2_beta;
7batch_normalization_2_batch_normalization_2_moving_mean?
;batch_normalization_2_batch_normalization_2_moving_variance5
1batch_normalization_2_batch_normalization_2_const
conv2d_2_conv2d_2_kernel5
1batch_normalization_3_batch_normalization_3_gamma4
0batch_normalization_3_batch_normalization_3_beta;
7batch_normalization_3_batch_normalization_3_moving_mean?
;batch_normalization_3_batch_normalization_3_moving_variance
conv2d_3_conv2d_3_kernel4
0batch_normalization_4_batch_normalization_4_beta;
7batch_normalization_4_batch_normalization_4_moving_mean?
;batch_normalization_4_batch_normalization_4_moving_variance5
1batch_normalization_4_batch_normalization_4_const:
6depthwise_conv2d_1_depthwise_conv2d_1_depthwise_kernel4
0batch_normalization_5_batch_normalization_5_beta;
7batch_normalization_5_batch_normalization_5_moving_mean?
;batch_normalization_5_batch_normalization_5_moving_variance5
1batch_normalization_5_batch_normalization_5_const
conv2d_4_conv2d_4_kernel5
1batch_normalization_6_batch_normalization_6_gamma4
0batch_normalization_6_batch_normalization_6_beta;
7batch_normalization_6_batch_normalization_6_moving_mean?
;batch_normalization_6_batch_normalization_6_moving_variance
conv2d_5_conv2d_5_kernel4
0batch_normalization_7_batch_normalization_7_beta;
7batch_normalization_7_batch_normalization_7_moving_mean?
;batch_normalization_7_batch_normalization_7_moving_variance5
1batch_normalization_7_batch_normalization_7_const:
6depthwise_conv2d_2_depthwise_conv2d_2_depthwise_kernel4
0batch_normalization_8_batch_normalization_8_beta;
7batch_normalization_8_batch_normalization_8_moving_mean?
;batch_normalization_8_batch_normalization_8_moving_variance5
1batch_normalization_8_batch_normalization_8_const
conv2d_6_conv2d_6_kernel5
1batch_normalization_9_batch_normalization_9_gamma4
0batch_normalization_9_batch_normalization_9_beta;
7batch_normalization_9_batch_normalization_9_moving_mean?
;batch_normalization_9_batch_normalization_9_moving_variance
conv2d_7_conv2d_7_kernel6
2batch_normalization_10_batch_normalization_10_beta=
9batch_normalization_10_batch_normalization_10_moving_meanA
=batch_normalization_10_batch_normalization_10_moving_variance7
3batch_normalization_10_batch_normalization_10_const:
6depthwise_conv2d_3_depthwise_conv2d_3_depthwise_kernel6
2batch_normalization_11_batch_normalization_11_beta=
9batch_normalization_11_batch_normalization_11_moving_meanA
=batch_normalization_11_batch_normalization_11_moving_variance7
3batch_normalization_11_batch_normalization_11_const
conv2d_8_conv2d_8_kernel7
3batch_normalization_12_batch_normalization_12_gamma6
2batch_normalization_12_batch_normalization_12_beta=
9batch_normalization_12_batch_normalization_12_moving_meanA
=batch_normalization_12_batch_normalization_12_moving_variance
dense_dense_kernel
dense_dense_bias
identityข+batch_normalization/StatefulPartitionedCallข-batch_normalization_1/StatefulPartitionedCallข.batch_normalization_10/StatefulPartitionedCallข.batch_normalization_11/StatefulPartitionedCallข.batch_normalization_12/StatefulPartitionedCallข-batch_normalization_2/StatefulPartitionedCallข-batch_normalization_3/StatefulPartitionedCallข-batch_normalization_4/StatefulPartitionedCallข-batch_normalization_5/StatefulPartitionedCallข-batch_normalization_6/StatefulPartitionedCallข-batch_normalization_7/StatefulPartitionedCallข-batch_normalization_8/StatefulPartitionedCallข-batch_normalization_9/StatefulPartitionedCallขconv2d/StatefulPartitionedCallข conv2d_1/StatefulPartitionedCallข conv2d_2/StatefulPartitionedCallข conv2d_3/StatefulPartitionedCallข conv2d_4/StatefulPartitionedCallข conv2d_5/StatefulPartitionedCallข conv2d_6/StatefulPartitionedCallข conv2d_7/StatefulPartitionedCallข conv2d_8/StatefulPartitionedCallขdense/StatefulPartitionedCallข(depthwise_conv2d/StatefulPartitionedCallข*depthwise_conv2d_1/StatefulPartitionedCallข*depthwise_conv2d_2/StatefulPartitionedCallข*depthwise_conv2d_3/StatefulPartitionedCall๛
&tf_op_layer_ExpandDims/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:1(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_22192(
&tf_op_layer_ExpandDims/PartitionedCallฆ
conv2d/StatefulPartitionedCallStatefulPartitionedCall/tf_op_layer_ExpandDims/PartitionedCall:output:0conv2d_conv2d_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_22342 
conv2d/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0,batch_normalization_batch_normalization_beta3batch_normalization_batch_normalization_moving_mean7batch_normalization_batch_normalization_moving_variance-batch_normalization_batch_normalization_const*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_22782-
+batch_normalization/StatefulPartitionedCall๖
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_re_lu_layer_call_and_return_conditional_losses_23072
re_lu/PartitionedCall
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0conv2d_1_conv2d_1_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_23222"
 conv2d_1/StatefulPartitionedCall
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:00batch_normalization_1_batch_normalization_1_beta7batch_normalization_1_batch_normalization_1_moving_mean;batch_normalization_1_batch_normalization_1_moving_variance1batch_normalization_1_batch_normalization_1_const*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_23652/
-batch_normalization_1/StatefulPartitionedCall?
re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_re_lu_1_layer_call_and_return_conditional_losses_23942
re_lu_1/PartitionedCallำ
(depthwise_conv2d/StatefulPartitionedCallStatefulPartitionedCall re_lu_1/PartitionedCall:output:02depthwise_conv2d_depthwise_conv2d_depthwise_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_depthwise_conv2d_layer_call_and_return_conditional_losses_11242*
(depthwise_conv2d/StatefulPartitionedCall
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall1depthwise_conv2d/StatefulPartitionedCall:output:00batch_normalization_2_batch_normalization_2_beta7batch_normalization_2_batch_normalization_2_moving_mean;batch_normalization_2_batch_normalization_2_moving_variance1batch_normalization_2_batch_normalization_2_const*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_24432/
-batch_normalization_2/StatefulPartitionedCall?
re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_re_lu_2_layer_call_and_return_conditional_losses_24722
re_lu_2/PartitionedCallก
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall re_lu_2/PartitionedCall:output:0conv2d_2_conv2d_2_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_24872"
 conv2d_2/StatefulPartitionedCall
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:01batch_normalization_3_batch_normalization_3_gamma0batch_normalization_3_batch_normalization_3_beta7batch_normalization_3_batch_normalization_3_moving_mean;batch_normalization_3_batch_normalization_3_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_25322/
-batch_normalization_3/StatefulPartitionedCall
add/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:06batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *F
fAR?
=__inference_add_layer_call_and_return_conditional_losses_25622
add/PartitionedCall
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0conv2d_3_conv2d_3_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_25782"
 conv2d_3/StatefulPartitionedCall
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:00batch_normalization_4_batch_normalization_4_beta7batch_normalization_4_batch_normalization_4_moving_mean;batch_normalization_4_batch_normalization_4_moving_variance1batch_normalization_4_batch_normalization_4_const*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_26212/
-batch_normalization_4/StatefulPartitionedCall?
re_lu_3/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_re_lu_3_layer_call_and_return_conditional_losses_26502
re_lu_3/PartitionedCall?
*depthwise_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall re_lu_3/PartitionedCall:output:06depthwise_conv2d_1_depthwise_conv2d_1_depthwise_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_depthwise_conv2d_1_layer_call_and_return_conditional_losses_14182,
*depthwise_conv2d_1/StatefulPartitionedCallก
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall3depthwise_conv2d_1/StatefulPartitionedCall:output:00batch_normalization_5_batch_normalization_5_beta7batch_normalization_5_batch_normalization_5_moving_mean;batch_normalization_5_batch_normalization_5_moving_variance1batch_normalization_5_batch_normalization_5_const*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_26992/
-batch_normalization_5/StatefulPartitionedCall?
re_lu_4/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_re_lu_4_layer_call_and_return_conditional_losses_27282
re_lu_4/PartitionedCallก
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:0conv2d_4_conv2d_4_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_27432"
 conv2d_4/StatefulPartitionedCall
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:01batch_normalization_6_batch_normalization_6_gamma0batch_normalization_6_batch_normalization_6_beta7batch_normalization_6_batch_normalization_6_moving_mean;batch_normalization_6_batch_normalization_6_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_27882/
-batch_normalization_6/StatefulPartitionedCallท
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0conv2d_5_conv2d_5_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_5_layer_call_and_return_conditional_losses_28192"
 conv2d_5/StatefulPartitionedCall
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:00batch_normalization_7_batch_normalization_7_beta7batch_normalization_7_batch_normalization_7_moving_mean;batch_normalization_7_batch_normalization_7_moving_variance1batch_normalization_7_batch_normalization_7_const*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_28622/
-batch_normalization_7/StatefulPartitionedCall?
re_lu_5/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_re_lu_5_layer_call_and_return_conditional_losses_28912
re_lu_5/PartitionedCall?
*depthwise_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall re_lu_5/PartitionedCall:output:06depthwise_conv2d_2_depthwise_conv2d_2_depthwise_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_depthwise_conv2d_2_layer_call_and_return_conditional_losses_17122,
*depthwise_conv2d_2/StatefulPartitionedCallก
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall3depthwise_conv2d_2/StatefulPartitionedCall:output:00batch_normalization_8_batch_normalization_8_beta7batch_normalization_8_batch_normalization_8_moving_mean;batch_normalization_8_batch_normalization_8_moving_variance1batch_normalization_8_batch_normalization_8_const*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_29402/
-batch_normalization_8/StatefulPartitionedCall?
re_lu_6/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_re_lu_6_layer_call_and_return_conditional_losses_29692
re_lu_6/PartitionedCallก
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall re_lu_6/PartitionedCall:output:0conv2d_6_conv2d_6_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_6_layer_call_and_return_conditional_losses_29842"
 conv2d_6/StatefulPartitionedCall
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:01batch_normalization_9_batch_normalization_9_gamma0batch_normalization_9_batch_normalization_9_beta7batch_normalization_9_batch_normalization_9_moving_mean;batch_normalization_9_batch_normalization_9_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_9_layer_call_and_return_conditional_losses_30292/
-batch_normalization_9/StatefulPartitionedCallท
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0conv2d_7_conv2d_7_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:`*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_7_layer_call_and_return_conditional_losses_30602"
 conv2d_7/StatefulPartitionedCallข
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:02batch_normalization_10_batch_normalization_10_beta9batch_normalization_10_batch_normalization_10_moving_mean=batch_normalization_10_batch_normalization_10_moving_variance3batch_normalization_10_batch_normalization_10_const*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:`*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_310320
.batch_normalization_10/StatefulPartitionedCall?
re_lu_7/PartitionedCallPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_re_lu_7_layer_call_and_return_conditional_losses_31322
re_lu_7/PartitionedCall?
*depthwise_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_7/PartitionedCall:output:06depthwise_conv2d_3_depthwise_conv2d_3_depthwise_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:`*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_depthwise_conv2d_3_layer_call_and_return_conditional_losses_20062,
*depthwise_conv2d_3/StatefulPartitionedCallฌ
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall3depthwise_conv2d_3/StatefulPartitionedCall:output:02batch_normalization_11_batch_normalization_11_beta9batch_normalization_11_batch_normalization_11_moving_mean=batch_normalization_11_batch_normalization_11_moving_variance3batch_normalization_11_batch_normalization_11_const*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:`*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_318120
.batch_normalization_11/StatefulPartitionedCall?
re_lu_8/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_re_lu_8_layer_call_and_return_conditional_losses_32102
re_lu_8/PartitionedCallก
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall re_lu_8/PartitionedCall:output:0conv2d_8_conv2d_8_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_8_layer_call_and_return_conditional_losses_32252"
 conv2d_8/StatefulPartitionedCallฃ
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:03batch_normalization_12_batch_normalization_12_gamma2batch_normalization_12_batch_normalization_12_beta9batch_normalization_12_batch_normalization_12_moving_mean=batch_normalization_12_batch_normalization_12_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_12_layer_call_and_return_conditional_losses_327020
.batch_normalization_12/StatefulPartitionedCallฒ
add_1/PartitionedCallPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:07batch_normalization_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_add_1_layer_call_and_return_conditional_losses_33002
add_1/PartitionedCall
(global_average_pooling2d/PartitionedCallPartitionedCalladd_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_22062*
(global_average_pooling2d/PartitionedCall๑
dropout/PartitionedCallPartitionedCall1global_average_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_33322
dropout/PartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_dense_kerneldense_dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_33552
dense/StatefulPartitionedCallํ	
IdentityIdentity&dense/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall/^batch_normalization_12/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall^dense/StatefulPartitionedCall)^depthwise_conv2d/StatefulPartitionedCall+^depthwise_conv2d_1/StatefulPartitionedCall+^depthwise_conv2d_2/StatefulPartitionedCall+^depthwise_conv2d_3/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*ม
_input_shapesฏ
ฌ:1(::::: :::::0:::::0::::::::::0:::::0::::::::::0:::::0::::::::::`:::::`:::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2T
(depthwise_conv2d/StatefulPartitionedCall(depthwise_conv2d/StatefulPartitionedCall2X
*depthwise_conv2d_1/StatefulPartitionedCall*depthwise_conv2d_1/StatefulPartitionedCall2X
*depthwise_conv2d_2/StatefulPartitionedCall*depthwise_conv2d_2/StatefulPartitionedCall2X
*depthwise_conv2d_3/StatefulPartitionedCall*depthwise_conv2d_3/StatefulPartitionedCall:S O
+
_output_shapes
:?????????1(
 
_user_specified_nameinputs: 

_output_shapes
: : 


_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0: (

_output_shapes
:0: -

_output_shapes
:0: 7

_output_shapes
:`: <

_output_shapes
:`
๐	

4__inference_batch_normalization_2_layer_call_fn_5526

inputs
batch_normalization_2_beta%
!batch_normalization_2_moving_mean)
%batch_normalization_2_moving_variance
batch_normalization_2_const
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_2_beta!batch_normalization_2_moving_mean%batch_normalization_2_moving_variancebatch_normalization_2_const*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????0*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_12092
StatefulPartitionedCallจ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:+???????????????????????????0::::022
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs: 

_output_shapes
:0
ษ

B__inference_conv2d_3_layer_call_and_return_conditional_losses_2578

inputs)
%conv2d_readvariableop_conv2d_3_kernel
identity
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_3_kernel*&
_output_shapes
: 0*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:0*
paddingSAME*
strides
2
Conv2Db
IdentityIdentityConv2D:output:0*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*)
_input_shapes
: ::N J
&
_output_shapes
: 
 
_user_specified_nameinputs
ฏ
]
A__inference_re_lu_4_layer_call_and_return_conditional_losses_2728

inputs
identityP
Relu6Relu6inputs*
T0*&
_output_shapes
:02
Relu6f
IdentityIdentityRelu6:activations:0*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*%
_input_shapes
:0:N J
&
_output_shapes
:0
 
_user_specified_nameinputs
ข
ไ1
F__inference_functional_1_layer_call_and_return_conditional_losses_4176
input_1.
*conv2d_conv2d_readvariableop_conv2d_kernel?
;batch_normalization_readvariableop_batch_normalization_betaW
Sbatch_normalization_fusedbatchnormv3_readvariableop_batch_normalization_moving_mean]
Ybatch_normalization_fusedbatchnormv3_readvariableop_1_batch_normalization_moving_varianceB
>batch_normalization_fusedbatchnormv3_batch_normalization_const2
.conv2d_1_conv2d_readvariableop_conv2d_1_kernelC
?batch_normalization_1_readvariableop_batch_normalization_1_beta[
Wbatch_normalization_1_fusedbatchnormv3_readvariableop_batch_normalization_1_moving_meana
]batch_normalization_1_fusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_varianceF
Bbatch_normalization_1_fusedbatchnormv3_batch_normalization_1_constO
Kdepthwise_conv2d_depthwise_readvariableop_depthwise_conv2d_depthwise_kernelC
?batch_normalization_2_readvariableop_batch_normalization_2_beta[
Wbatch_normalization_2_fusedbatchnormv3_readvariableop_batch_normalization_2_moving_meana
]batch_normalization_2_fusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_varianceF
Bbatch_normalization_2_fusedbatchnormv3_batch_normalization_2_const2
.conv2d_2_conv2d_readvariableop_conv2d_2_kernelD
@batch_normalization_3_readvariableop_batch_normalization_3_gammaE
Abatch_normalization_3_readvariableop_1_batch_normalization_3_beta[
Wbatch_normalization_3_fusedbatchnormv3_readvariableop_batch_normalization_3_moving_meana
]batch_normalization_3_fusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance2
.conv2d_3_conv2d_readvariableop_conv2d_3_kernelC
?batch_normalization_4_readvariableop_batch_normalization_4_beta[
Wbatch_normalization_4_fusedbatchnormv3_readvariableop_batch_normalization_4_moving_meana
]batch_normalization_4_fusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_varianceF
Bbatch_normalization_4_fusedbatchnormv3_batch_normalization_4_constS
Odepthwise_conv2d_1_depthwise_readvariableop_depthwise_conv2d_1_depthwise_kernelC
?batch_normalization_5_readvariableop_batch_normalization_5_beta[
Wbatch_normalization_5_fusedbatchnormv3_readvariableop_batch_normalization_5_moving_meana
]batch_normalization_5_fusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_varianceF
Bbatch_normalization_5_fusedbatchnormv3_batch_normalization_5_const2
.conv2d_4_conv2d_readvariableop_conv2d_4_kernelD
@batch_normalization_6_readvariableop_batch_normalization_6_gammaE
Abatch_normalization_6_readvariableop_1_batch_normalization_6_beta[
Wbatch_normalization_6_fusedbatchnormv3_readvariableop_batch_normalization_6_moving_meana
]batch_normalization_6_fusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance2
.conv2d_5_conv2d_readvariableop_conv2d_5_kernelC
?batch_normalization_7_readvariableop_batch_normalization_7_beta[
Wbatch_normalization_7_fusedbatchnormv3_readvariableop_batch_normalization_7_moving_meana
]batch_normalization_7_fusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_varianceF
Bbatch_normalization_7_fusedbatchnormv3_batch_normalization_7_constS
Odepthwise_conv2d_2_depthwise_readvariableop_depthwise_conv2d_2_depthwise_kernelC
?batch_normalization_8_readvariableop_batch_normalization_8_beta[
Wbatch_normalization_8_fusedbatchnormv3_readvariableop_batch_normalization_8_moving_meana
]batch_normalization_8_fusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_varianceF
Bbatch_normalization_8_fusedbatchnormv3_batch_normalization_8_const2
.conv2d_6_conv2d_readvariableop_conv2d_6_kernelD
@batch_normalization_9_readvariableop_batch_normalization_9_gammaE
Abatch_normalization_9_readvariableop_1_batch_normalization_9_beta[
Wbatch_normalization_9_fusedbatchnormv3_readvariableop_batch_normalization_9_moving_meana
]batch_normalization_9_fusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance2
.conv2d_7_conv2d_readvariableop_conv2d_7_kernelE
Abatch_normalization_10_readvariableop_batch_normalization_10_beta]
Ybatch_normalization_10_fusedbatchnormv3_readvariableop_batch_normalization_10_moving_meanc
_batch_normalization_10_fusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_varianceH
Dbatch_normalization_10_fusedbatchnormv3_batch_normalization_10_constS
Odepthwise_conv2d_3_depthwise_readvariableop_depthwise_conv2d_3_depthwise_kernelE
Abatch_normalization_11_readvariableop_batch_normalization_11_beta]
Ybatch_normalization_11_fusedbatchnormv3_readvariableop_batch_normalization_11_moving_meanc
_batch_normalization_11_fusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_varianceH
Dbatch_normalization_11_fusedbatchnormv3_batch_normalization_11_const2
.conv2d_8_conv2d_readvariableop_conv2d_8_kernelF
Bbatch_normalization_12_readvariableop_batch_normalization_12_gammaG
Cbatch_normalization_12_readvariableop_1_batch_normalization_12_beta]
Ybatch_normalization_12_fusedbatchnormv3_readvariableop_batch_normalization_12_moving_meanc
_batch_normalization_12_fusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance,
(dense_matmul_readvariableop_dense_kernel+
'dense_biasadd_readvariableop_dense_bias
identityข"batch_normalization/AssignNewValueข$batch_normalization/AssignNewValue_1ข$batch_normalization_1/AssignNewValueข&batch_normalization_1/AssignNewValue_1ข%batch_normalization_10/AssignNewValueข'batch_normalization_10/AssignNewValue_1ข%batch_normalization_11/AssignNewValueข'batch_normalization_11/AssignNewValue_1ข%batch_normalization_12/AssignNewValueข'batch_normalization_12/AssignNewValue_1ข$batch_normalization_2/AssignNewValueข&batch_normalization_2/AssignNewValue_1ข$batch_normalization_3/AssignNewValueข&batch_normalization_3/AssignNewValue_1ข$batch_normalization_4/AssignNewValueข&batch_normalization_4/AssignNewValue_1ข$batch_normalization_5/AssignNewValueข&batch_normalization_5/AssignNewValue_1ข$batch_normalization_6/AssignNewValueข&batch_normalization_6/AssignNewValue_1ข$batch_normalization_7/AssignNewValueข&batch_normalization_7/AssignNewValue_1ข$batch_normalization_8/AssignNewValueข&batch_normalization_8/AssignNewValue_1ข$batch_normalization_9/AssignNewValueข&batch_normalization_9/AssignNewValue_1
%tf_op_layer_ExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%tf_op_layer_ExpandDims/ExpandDims/dimอ
!tf_op_layer_ExpandDims/ExpandDims
ExpandDimsinput_1.tf_op_layer_ExpandDims/ExpandDims/dim:output:0*
T0*
_cloned(*&
_output_shapes
:1(2#
!tf_op_layer_ExpandDims/ExpandDimsฏ
conv2d/Conv2D/ReadVariableOpReadVariableOp*conv2d_conv2d_readvariableop_conv2d_kernel*&
_output_shapes
:( *
dtype02
conv2d/Conv2D/ReadVariableOpิ
conv2d/Conv2DConv2D*tf_op_layer_ExpandDims/ExpandDims:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
2
conv2d/Conv2Dภ
"batch_normalization/ReadVariableOpReadVariableOp;batch_normalization_readvariableop_batch_normalization_beta*
_output_shapes
: *
dtype02$
"batch_normalization/ReadVariableOp๚
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpSbatch_normalization_fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
_output_shapes
: *
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpYbatch_normalization_fusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
_output_shapes
: *
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1้
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/Conv2D:output:0>batch_normalization_fusedbatchnormv3_batch_normalization_const*batch_normalization/ReadVariableOp:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
exponential_avg_factor%
ื#<2&
$batch_normalization/FusedBatchNormV3ฅ
"batch_normalization/AssignNewValueAssignVariableOpSbatch_normalization_fusedbatchnormv3_readvariableop_batch_normalization_moving_mean1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*f
_class\
ZXloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp/batch_normalization/moving_mean*
_output_shapes
 *
dtype02$
"batch_normalization/AssignNewValueป
$batch_normalization/AssignNewValue_1AssignVariableOpYbatch_normalization_fusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*l
_classb
`^loc:@batch_normalization/FusedBatchNormV3/ReadVariableOp_1/batch_normalization/moving_variance*
_output_shapes
 *
dtype02&
$batch_normalization/AssignNewValue_1~
re_lu/Relu6Relu6(batch_normalization/FusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2
re_lu/Relu6ท
conv2d_1/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_conv2d_readvariableop_conv2d_1_kernel*&
_output_shapes
: 0*
dtype02 
conv2d_1/Conv2D/ReadVariableOpศ
conv2d_1/Conv2DConv2Dre_lu/Relu6:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:0*
paddingSAME*
strides
2
conv2d_1/Conv2Dศ
$batch_normalization_1/ReadVariableOpReadVariableOp?batch_normalization_1_readvariableop_batch_normalization_1_beta*
_output_shapes
:0*
dtype02&
$batch_normalization_1/ReadVariableOp
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_1_fusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
:0*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_1_fusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
_output_shapes
:0*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1๙
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/Conv2D:output:0Bbatch_normalization_1_fusedbatchnormv3_batch_normalization_1_const,batch_normalization_1/ReadVariableOp:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
ื#<2(
&batch_normalization_1/FusedBatchNormV3ต
$batch_normalization_1/AssignNewValueAssignVariableOpWbatch_normalization_1_fusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*j
_class`
^\loc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp/batch_normalization_1/moving_mean*
_output_shapes
 *
dtype02&
$batch_normalization_1/AssignNewValueห
&batch_normalization_1/AssignNewValue_1AssignVariableOp]batch_normalization_1_fusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*p
_classf
dbloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_1/moving_variance*
_output_shapes
 *
dtype02(
&batch_normalization_1/AssignNewValue_1
re_lu_1/Relu6Relu6*batch_normalization_1/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:02
re_lu_1/Relu6๊
)depthwise_conv2d/depthwise/ReadVariableOpReadVariableOpKdepthwise_conv2d_depthwise_readvariableop_depthwise_conv2d_depthwise_kernel*&
_output_shapes
:0*
dtype02+
)depthwise_conv2d/depthwise/ReadVariableOp
 depthwise_conv2d/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      0      2"
 depthwise_conv2d/depthwise/Shapeฅ
(depthwise_conv2d/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2*
(depthwise_conv2d/depthwise/dilation_rate๚
depthwise_conv2d/depthwiseDepthwiseConv2dNativere_lu_1/Relu6:activations:01depthwise_conv2d/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:0*
paddingSAME*
strides
2
depthwise_conv2d/depthwiseศ
$batch_normalization_2/ReadVariableOpReadVariableOp?batch_normalization_2_readvariableop_batch_normalization_2_beta*
_output_shapes
:0*
dtype02&
$batch_normalization_2/ReadVariableOp
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_2_fusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
:0*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_2_fusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
_output_shapes
:0*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3#depthwise_conv2d/depthwise:output:0Bbatch_normalization_2_fusedbatchnormv3_batch_normalization_2_const,batch_normalization_2/ReadVariableOp:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
ื#<2(
&batch_normalization_2/FusedBatchNormV3ต
$batch_normalization_2/AssignNewValueAssignVariableOpWbatch_normalization_2_fusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*j
_class`
^\loc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp/batch_normalization_2/moving_mean*
_output_shapes
 *
dtype02&
$batch_normalization_2/AssignNewValueห
&batch_normalization_2/AssignNewValue_1AssignVariableOp]batch_normalization_2_fusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*p
_classf
dbloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_2/moving_variance*
_output_shapes
 *
dtype02(
&batch_normalization_2/AssignNewValue_1
re_lu_2/Relu6Relu6*batch_normalization_2/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:02
re_lu_2/Relu6ท
conv2d_2/Conv2D/ReadVariableOpReadVariableOp.conv2d_2_conv2d_readvariableop_conv2d_2_kernel*&
_output_shapes
:0 *
dtype02 
conv2d_2/Conv2D/ReadVariableOpส
conv2d_2/Conv2DConv2Dre_lu_2/Relu6:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingSAME*
strides
2
conv2d_2/Conv2Dษ
$batch_normalization_3/ReadVariableOpReadVariableOp@batch_normalization_3_readvariableop_batch_normalization_3_gamma*
_output_shapes
: *
dtype02&
$batch_normalization_3/ReadVariableOpฮ
&batch_normalization_3/ReadVariableOp_1ReadVariableOpAbatch_normalization_3_readvariableop_1_batch_normalization_3_beta*
_output_shapes
: *
dtype02(
&batch_normalization_3/ReadVariableOp_1
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_3_fusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
_output_shapes
: *
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_3_fusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
_output_shapes
: *
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ๅ
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_2/Conv2D:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
exponential_avg_factor%
ื#<2(
&batch_normalization_3/FusedBatchNormV3ต
$batch_normalization_3/AssignNewValueAssignVariableOpWbatch_normalization_3_fusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*j
_class`
^\loc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp/batch_normalization_3/moving_mean*
_output_shapes
 *
dtype02&
$batch_normalization_3/AssignNewValueห
&batch_normalization_3/AssignNewValue_1AssignVariableOp]batch_normalization_3_fusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance7batch_normalization_3/FusedBatchNormV3:batch_variance:08^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*p
_classf
dbloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_3/moving_variance*
_output_shapes
 *
dtype02(
&batch_normalization_3/AssignNewValue_1
add/addAddV2re_lu/Relu6:activations:0*batch_normalization_3/FusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2	
add/addท
conv2d_3/Conv2D/ReadVariableOpReadVariableOp.conv2d_3_conv2d_readvariableop_conv2d_3_kernel*&
_output_shapes
: 0*
dtype02 
conv2d_3/Conv2D/ReadVariableOpบ
conv2d_3/Conv2DConv2Dadd/add:z:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:0*
paddingSAME*
strides
2
conv2d_3/Conv2Dศ
$batch_normalization_4/ReadVariableOpReadVariableOp?batch_normalization_4_readvariableop_batch_normalization_4_beta*
_output_shapes
:0*
dtype02&
$batch_normalization_4/ReadVariableOp
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_4_fusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
_output_shapes
:0*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_4_fusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
_output_shapes
:0*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1๙
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_3/Conv2D:output:0Bbatch_normalization_4_fusedbatchnormv3_batch_normalization_4_const,batch_normalization_4/ReadVariableOp:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
ื#<2(
&batch_normalization_4/FusedBatchNormV3ต
$batch_normalization_4/AssignNewValueAssignVariableOpWbatch_normalization_4_fusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean3batch_normalization_4/FusedBatchNormV3:batch_mean:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp*j
_class`
^\loc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp/batch_normalization_4/moving_mean*
_output_shapes
 *
dtype02&
$batch_normalization_4/AssignNewValueห
&batch_normalization_4/AssignNewValue_1AssignVariableOp]batch_normalization_4_fusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance7batch_normalization_4/FusedBatchNormV3:batch_variance:08^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*p
_classf
dbloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_4/moving_variance*
_output_shapes
 *
dtype02(
&batch_normalization_4/AssignNewValue_1
re_lu_3/Relu6Relu6*batch_normalization_4/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:02
re_lu_3/Relu6๒
+depthwise_conv2d_1/depthwise/ReadVariableOpReadVariableOpOdepthwise_conv2d_1_depthwise_readvariableop_depthwise_conv2d_1_depthwise_kernel*&
_output_shapes
:0*
dtype02-
+depthwise_conv2d_1/depthwise/ReadVariableOpก
"depthwise_conv2d_1/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      0      2$
"depthwise_conv2d_1/depthwise/Shapeฉ
*depthwise_conv2d_1/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2,
*depthwise_conv2d_1/depthwise/dilation_rate
depthwise_conv2d_1/depthwiseDepthwiseConv2dNativere_lu_3/Relu6:activations:03depthwise_conv2d_1/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:0*
paddingSAME*
strides
2
depthwise_conv2d_1/depthwiseศ
$batch_normalization_5/ReadVariableOpReadVariableOp?batch_normalization_5_readvariableop_batch_normalization_5_beta*
_output_shapes
:0*
dtype02&
$batch_normalization_5/ReadVariableOp
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_5_fusedbatchnormv3_readvariableop_batch_normalization_5_moving_mean*
_output_shapes
:0*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_5_fusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance*
_output_shapes
:0*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3%depthwise_conv2d_1/depthwise:output:0Bbatch_normalization_5_fusedbatchnormv3_batch_normalization_5_const,batch_normalization_5/ReadVariableOp:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
ื#<2(
&batch_normalization_5/FusedBatchNormV3ต
$batch_normalization_5/AssignNewValueAssignVariableOpWbatch_normalization_5_fusedbatchnormv3_readvariableop_batch_normalization_5_moving_mean3batch_normalization_5/FusedBatchNormV3:batch_mean:06^batch_normalization_5/FusedBatchNormV3/ReadVariableOp*j
_class`
^\loc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp/batch_normalization_5/moving_mean*
_output_shapes
 *
dtype02&
$batch_normalization_5/AssignNewValueห
&batch_normalization_5/AssignNewValue_1AssignVariableOp]batch_normalization_5_fusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance7batch_normalization_5/FusedBatchNormV3:batch_variance:08^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*p
_classf
dbloc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_5/moving_variance*
_output_shapes
 *
dtype02(
&batch_normalization_5/AssignNewValue_1
re_lu_4/Relu6Relu6*batch_normalization_5/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:02
re_lu_4/Relu6ท
conv2d_4/Conv2D/ReadVariableOpReadVariableOp.conv2d_4_conv2d_readvariableop_conv2d_4_kernel*&
_output_shapes
:0 *
dtype02 
conv2d_4/Conv2D/ReadVariableOpส
conv2d_4/Conv2DConv2Dre_lu_4/Relu6:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingSAME*
strides
2
conv2d_4/Conv2Dษ
$batch_normalization_6/ReadVariableOpReadVariableOp@batch_normalization_6_readvariableop_batch_normalization_6_gamma*
_output_shapes
: *
dtype02&
$batch_normalization_6/ReadVariableOpฮ
&batch_normalization_6/ReadVariableOp_1ReadVariableOpAbatch_normalization_6_readvariableop_1_batch_normalization_6_beta*
_output_shapes
: *
dtype02(
&batch_normalization_6/ReadVariableOp_1
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_6_fusedbatchnormv3_readvariableop_batch_normalization_6_moving_mean*
_output_shapes
: *
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_6_fusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance*
_output_shapes
: *
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ๅ
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv2d_4/Conv2D:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
exponential_avg_factor%
ื#<2(
&batch_normalization_6/FusedBatchNormV3ต
$batch_normalization_6/AssignNewValueAssignVariableOpWbatch_normalization_6_fusedbatchnormv3_readvariableop_batch_normalization_6_moving_mean3batch_normalization_6/FusedBatchNormV3:batch_mean:06^batch_normalization_6/FusedBatchNormV3/ReadVariableOp*j
_class`
^\loc:@batch_normalization_6/FusedBatchNormV3/ReadVariableOp/batch_normalization_6/moving_mean*
_output_shapes
 *
dtype02&
$batch_normalization_6/AssignNewValueห
&batch_normalization_6/AssignNewValue_1AssignVariableOp]batch_normalization_6_fusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance7batch_normalization_6/FusedBatchNormV3:batch_variance:08^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1*p
_classf
dbloc:@batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_6/moving_variance*
_output_shapes
 *
dtype02(
&batch_normalization_6/AssignNewValue_1ท
conv2d_5/Conv2D/ReadVariableOpReadVariableOp.conv2d_5_conv2d_readvariableop_conv2d_5_kernel*&
_output_shapes
: 0*
dtype02 
conv2d_5/Conv2D/ReadVariableOpู
conv2d_5/Conv2DConv2D*batch_normalization_6/FusedBatchNormV3:y:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:0*
paddingSAME*
strides
2
conv2d_5/Conv2Dศ
$batch_normalization_7/ReadVariableOpReadVariableOp?batch_normalization_7_readvariableop_batch_normalization_7_beta*
_output_shapes
:0*
dtype02&
$batch_normalization_7/ReadVariableOp
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_7_fusedbatchnormv3_readvariableop_batch_normalization_7_moving_mean*
_output_shapes
:0*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_7_fusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance*
_output_shapes
:0*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1๙
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3conv2d_5/Conv2D:output:0Bbatch_normalization_7_fusedbatchnormv3_batch_normalization_7_const,batch_normalization_7/ReadVariableOp:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
ื#<2(
&batch_normalization_7/FusedBatchNormV3ต
$batch_normalization_7/AssignNewValueAssignVariableOpWbatch_normalization_7_fusedbatchnormv3_readvariableop_batch_normalization_7_moving_mean3batch_normalization_7/FusedBatchNormV3:batch_mean:06^batch_normalization_7/FusedBatchNormV3/ReadVariableOp*j
_class`
^\loc:@batch_normalization_7/FusedBatchNormV3/ReadVariableOp/batch_normalization_7/moving_mean*
_output_shapes
 *
dtype02&
$batch_normalization_7/AssignNewValueห
&batch_normalization_7/AssignNewValue_1AssignVariableOp]batch_normalization_7_fusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance7batch_normalization_7/FusedBatchNormV3:batch_variance:08^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1*p
_classf
dbloc:@batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_7/moving_variance*
_output_shapes
 *
dtype02(
&batch_normalization_7/AssignNewValue_1
re_lu_5/Relu6Relu6*batch_normalization_7/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:02
re_lu_5/Relu6๒
+depthwise_conv2d_2/depthwise/ReadVariableOpReadVariableOpOdepthwise_conv2d_2_depthwise_readvariableop_depthwise_conv2d_2_depthwise_kernel*&
_output_shapes
:0*
dtype02-
+depthwise_conv2d_2/depthwise/ReadVariableOpก
"depthwise_conv2d_2/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      0      2$
"depthwise_conv2d_2/depthwise/Shapeฉ
*depthwise_conv2d_2/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2,
*depthwise_conv2d_2/depthwise/dilation_rate
depthwise_conv2d_2/depthwiseDepthwiseConv2dNativere_lu_5/Relu6:activations:03depthwise_conv2d_2/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:0*
paddingSAME*
strides
2
depthwise_conv2d_2/depthwiseศ
$batch_normalization_8/ReadVariableOpReadVariableOp?batch_normalization_8_readvariableop_batch_normalization_8_beta*
_output_shapes
:0*
dtype02&
$batch_normalization_8/ReadVariableOp
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_8_fusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean*
_output_shapes
:0*
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_8_fusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance*
_output_shapes
:0*
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3%depthwise_conv2d_2/depthwise:output:0Bbatch_normalization_8_fusedbatchnormv3_batch_normalization_8_const,batch_normalization_8/ReadVariableOp:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
ื#<2(
&batch_normalization_8/FusedBatchNormV3ต
$batch_normalization_8/AssignNewValueAssignVariableOpWbatch_normalization_8_fusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean3batch_normalization_8/FusedBatchNormV3:batch_mean:06^batch_normalization_8/FusedBatchNormV3/ReadVariableOp*j
_class`
^\loc:@batch_normalization_8/FusedBatchNormV3/ReadVariableOp/batch_normalization_8/moving_mean*
_output_shapes
 *
dtype02&
$batch_normalization_8/AssignNewValueห
&batch_normalization_8/AssignNewValue_1AssignVariableOp]batch_normalization_8_fusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance7batch_normalization_8/FusedBatchNormV3:batch_variance:08^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1*p
_classf
dbloc:@batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_8/moving_variance*
_output_shapes
 *
dtype02(
&batch_normalization_8/AssignNewValue_1
re_lu_6/Relu6Relu6*batch_normalization_8/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:02
re_lu_6/Relu6ท
conv2d_6/Conv2D/ReadVariableOpReadVariableOp.conv2d_6_conv2d_readvariableop_conv2d_6_kernel*&
_output_shapes
:0@*
dtype02 
conv2d_6/Conv2D/ReadVariableOpส
conv2d_6/Conv2DConv2Dre_lu_6/Relu6:activations:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingSAME*
strides
2
conv2d_6/Conv2Dษ
$batch_normalization_9/ReadVariableOpReadVariableOp@batch_normalization_9_readvariableop_batch_normalization_9_gamma*
_output_shapes
:@*
dtype02&
$batch_normalization_9/ReadVariableOpฮ
&batch_normalization_9/ReadVariableOp_1ReadVariableOpAbatch_normalization_9_readvariableop_1_batch_normalization_9_beta*
_output_shapes
:@*
dtype02(
&batch_normalization_9/ReadVariableOp_1
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_9_fusedbatchnormv3_readvariableop_batch_normalization_9_moving_mean*
_output_shapes
:@*
dtype027
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_9_fusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance*
_output_shapes
:@*
dtype029
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ๅ
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3conv2d_6/Conv2D:output:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
ื#<2(
&batch_normalization_9/FusedBatchNormV3ต
$batch_normalization_9/AssignNewValueAssignVariableOpWbatch_normalization_9_fusedbatchnormv3_readvariableop_batch_normalization_9_moving_mean3batch_normalization_9/FusedBatchNormV3:batch_mean:06^batch_normalization_9/FusedBatchNormV3/ReadVariableOp*j
_class`
^\loc:@batch_normalization_9/FusedBatchNormV3/ReadVariableOp/batch_normalization_9/moving_mean*
_output_shapes
 *
dtype02&
$batch_normalization_9/AssignNewValueห
&batch_normalization_9/AssignNewValue_1AssignVariableOp]batch_normalization_9_fusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance7batch_normalization_9/FusedBatchNormV3:batch_variance:08^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1*p
_classf
dbloc:@batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_9/moving_variance*
_output_shapes
 *
dtype02(
&batch_normalization_9/AssignNewValue_1ท
conv2d_7/Conv2D/ReadVariableOpReadVariableOp.conv2d_7_conv2d_readvariableop_conv2d_7_kernel*&
_output_shapes
:@`*
dtype02 
conv2d_7/Conv2D/ReadVariableOpู
conv2d_7/Conv2DConv2D*batch_normalization_9/FusedBatchNormV3:y:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:`*
paddingSAME*
strides
2
conv2d_7/Conv2Dฬ
%batch_normalization_10/ReadVariableOpReadVariableOpAbatch_normalization_10_readvariableop_batch_normalization_10_beta*
_output_shapes
:`*
dtype02'
%batch_normalization_10/ReadVariableOp
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOpYbatch_normalization_10_fusedbatchnormv3_readvariableop_batch_normalization_10_moving_mean*
_output_shapes
:`*
dtype028
6batch_normalization_10/FusedBatchNormV3/ReadVariableOp
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp_batch_normalization_10_fusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance*
_output_shapes
:`*
dtype02:
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1
'batch_normalization_10/FusedBatchNormV3FusedBatchNormV3conv2d_7/Conv2D:output:0Dbatch_normalization_10_fusedbatchnormv3_batch_normalization_10_const-batch_normalization_10/ReadVariableOp:value:0>batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:`:`:`:`:`:*
epsilon%o:*
exponential_avg_factor%
ื#<2)
'batch_normalization_10/FusedBatchNormV3ฝ
%batch_normalization_10/AssignNewValueAssignVariableOpYbatch_normalization_10_fusedbatchnormv3_readvariableop_batch_normalization_10_moving_mean4batch_normalization_10/FusedBatchNormV3:batch_mean:07^batch_normalization_10/FusedBatchNormV3/ReadVariableOp*l
_classb
`^loc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp/batch_normalization_10/moving_mean*
_output_shapes
 *
dtype02'
%batch_normalization_10/AssignNewValueำ
'batch_normalization_10/AssignNewValue_1AssignVariableOp_batch_normalization_10_fusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance8batch_normalization_10/FusedBatchNormV3:batch_variance:09^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1*r
_classh
fdloc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_10/moving_variance*
_output_shapes
 *
dtype02)
'batch_normalization_10/AssignNewValue_1
re_lu_7/Relu6Relu6+batch_normalization_10/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:`2
re_lu_7/Relu6๒
+depthwise_conv2d_3/depthwise/ReadVariableOpReadVariableOpOdepthwise_conv2d_3_depthwise_readvariableop_depthwise_conv2d_3_depthwise_kernel*&
_output_shapes
:`*
dtype02-
+depthwise_conv2d_3/depthwise/ReadVariableOpก
"depthwise_conv2d_3/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      `      2$
"depthwise_conv2d_3/depthwise/Shapeฉ
*depthwise_conv2d_3/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2,
*depthwise_conv2d_3/depthwise/dilation_rate
depthwise_conv2d_3/depthwiseDepthwiseConv2dNativere_lu_7/Relu6:activations:03depthwise_conv2d_3/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:`*
paddingSAME*
strides
2
depthwise_conv2d_3/depthwiseฬ
%batch_normalization_11/ReadVariableOpReadVariableOpAbatch_normalization_11_readvariableop_batch_normalization_11_beta*
_output_shapes
:`*
dtype02'
%batch_normalization_11/ReadVariableOp
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOpYbatch_normalization_11_fusedbatchnormv3_readvariableop_batch_normalization_11_moving_mean*
_output_shapes
:`*
dtype028
6batch_normalization_11/FusedBatchNormV3/ReadVariableOp
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp_batch_normalization_11_fusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance*
_output_shapes
:`*
dtype02:
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1
'batch_normalization_11/FusedBatchNormV3FusedBatchNormV3%depthwise_conv2d_3/depthwise:output:0Dbatch_normalization_11_fusedbatchnormv3_batch_normalization_11_const-batch_normalization_11/ReadVariableOp:value:0>batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:`:`:`:`:`:*
epsilon%o:*
exponential_avg_factor%
ื#<2)
'batch_normalization_11/FusedBatchNormV3ฝ
%batch_normalization_11/AssignNewValueAssignVariableOpYbatch_normalization_11_fusedbatchnormv3_readvariableop_batch_normalization_11_moving_mean4batch_normalization_11/FusedBatchNormV3:batch_mean:07^batch_normalization_11/FusedBatchNormV3/ReadVariableOp*l
_classb
`^loc:@batch_normalization_11/FusedBatchNormV3/ReadVariableOp/batch_normalization_11/moving_mean*
_output_shapes
 *
dtype02'
%batch_normalization_11/AssignNewValueำ
'batch_normalization_11/AssignNewValue_1AssignVariableOp_batch_normalization_11_fusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance8batch_normalization_11/FusedBatchNormV3:batch_variance:09^batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1*r
_classh
fdloc:@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_11/moving_variance*
_output_shapes
 *
dtype02)
'batch_normalization_11/AssignNewValue_1
re_lu_8/Relu6Relu6+batch_normalization_11/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:`2
re_lu_8/Relu6ท
conv2d_8/Conv2D/ReadVariableOpReadVariableOp.conv2d_8_conv2d_readvariableop_conv2d_8_kernel*&
_output_shapes
:`@*
dtype02 
conv2d_8/Conv2D/ReadVariableOpส
conv2d_8/Conv2DConv2Dre_lu_8/Relu6:activations:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingSAME*
strides
2
conv2d_8/Conv2Dอ
%batch_normalization_12/ReadVariableOpReadVariableOpBbatch_normalization_12_readvariableop_batch_normalization_12_gamma*
_output_shapes
:@*
dtype02'
%batch_normalization_12/ReadVariableOpา
'batch_normalization_12/ReadVariableOp_1ReadVariableOpCbatch_normalization_12_readvariableop_1_batch_normalization_12_beta*
_output_shapes
:@*
dtype02)
'batch_normalization_12/ReadVariableOp_1
6batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOpYbatch_normalization_12_fusedbatchnormv3_readvariableop_batch_normalization_12_moving_mean*
_output_shapes
:@*
dtype028
6batch_normalization_12/FusedBatchNormV3/ReadVariableOp
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp_batch_normalization_12_fusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance*
_output_shapes
:@*
dtype02:
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1๋
'batch_normalization_12/FusedBatchNormV3FusedBatchNormV3conv2d_8/Conv2D:output:0-batch_normalization_12/ReadVariableOp:value:0/batch_normalization_12/ReadVariableOp_1:value:0>batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
ื#<2)
'batch_normalization_12/FusedBatchNormV3ฝ
%batch_normalization_12/AssignNewValueAssignVariableOpYbatch_normalization_12_fusedbatchnormv3_readvariableop_batch_normalization_12_moving_mean4batch_normalization_12/FusedBatchNormV3:batch_mean:07^batch_normalization_12/FusedBatchNormV3/ReadVariableOp*l
_classb
`^loc:@batch_normalization_12/FusedBatchNormV3/ReadVariableOp/batch_normalization_12/moving_mean*
_output_shapes
 *
dtype02'
%batch_normalization_12/AssignNewValueำ
'batch_normalization_12/AssignNewValue_1AssignVariableOp_batch_normalization_12_fusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance8batch_normalization_12/FusedBatchNormV3:batch_variance:09^batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1*r
_classh
fdloc:@batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_12/moving_variance*
_output_shapes
 *
dtype02)
'batch_normalization_12/AssignNewValue_1ฉ
	add_1/addAddV2*batch_normalization_9/FusedBatchNormV3:y:0+batch_normalization_12/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:@2
	add_1/addณ
/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      21
/global_average_pooling2d/Mean/reduction_indicesธ
global_average_pooling2d/MeanMeanadd_1/add:z:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*
_output_shapes

:@2
global_average_pooling2d/Means
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/dropout/Constข
dropout/dropout/MulMul&global_average_pooling2d/Mean:output:0dropout/dropout/Const:output:0*
T0*
_output_shapes

:@2
dropout/dropout/Mul
dropout/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2
dropout/dropout/Shapeร
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*
_output_shapes

:@*
dtype02.
,dropout/dropout/random_uniform/RandomUniform
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *อฬL>2 
dropout/dropout/GreaterEqual/yี
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes

:@2
dropout/dropout/GreaterEqual
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:@2
dropout/dropout/Cast
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*
_output_shapes

:@2
dropout/dropout/Mul_1ฃ
dense/MatMul/ReadVariableOpReadVariableOp(dense_matmul_readvariableop_dense_kernel*
_output_shapes

:@*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMuldropout/dropout/Mul_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp'dense_biasadd_readvariableop_dense_bias*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense/BiasAdd๓
IdentityIdentitydense/BiasAdd:output:0#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_1&^batch_normalization_10/AssignNewValue(^batch_normalization_10/AssignNewValue_1&^batch_normalization_11/AssignNewValue(^batch_normalization_11/AssignNewValue_1&^batch_normalization_12/AssignNewValue(^batch_normalization_12/AssignNewValue_1%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_1%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_1%^batch_normalization_4/AssignNewValue'^batch_normalization_4/AssignNewValue_1%^batch_normalization_5/AssignNewValue'^batch_normalization_5/AssignNewValue_1%^batch_normalization_6/AssignNewValue'^batch_normalization_6/AssignNewValue_1%^batch_normalization_7/AssignNewValue'^batch_normalization_7/AssignNewValue_1%^batch_normalization_8/AssignNewValue'^batch_normalization_8/AssignNewValue_1%^batch_normalization_9/AssignNewValue'^batch_normalization_9/AssignNewValue_1*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*ม
_input_shapesฏ
ฌ:1(::::: :::::0:::::0::::::::::0:::::0::::::::::0:::::0::::::::::`:::::`:::::::2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12N
%batch_normalization_10/AssignNewValue%batch_normalization_10/AssignNewValue2R
'batch_normalization_10/AssignNewValue_1'batch_normalization_10/AssignNewValue_12N
%batch_normalization_11/AssignNewValue%batch_normalization_11/AssignNewValue2R
'batch_normalization_11/AssignNewValue_1'batch_normalization_11/AssignNewValue_12N
%batch_normalization_12/AssignNewValue%batch_normalization_12/AssignNewValue2R
'batch_normalization_12/AssignNewValue_1'batch_normalization_12/AssignNewValue_12L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_12L
$batch_normalization_3/AssignNewValue$batch_normalization_3/AssignNewValue2P
&batch_normalization_3/AssignNewValue_1&batch_normalization_3/AssignNewValue_12L
$batch_normalization_4/AssignNewValue$batch_normalization_4/AssignNewValue2P
&batch_normalization_4/AssignNewValue_1&batch_normalization_4/AssignNewValue_12L
$batch_normalization_5/AssignNewValue$batch_normalization_5/AssignNewValue2P
&batch_normalization_5/AssignNewValue_1&batch_normalization_5/AssignNewValue_12L
$batch_normalization_6/AssignNewValue$batch_normalization_6/AssignNewValue2P
&batch_normalization_6/AssignNewValue_1&batch_normalization_6/AssignNewValue_12L
$batch_normalization_7/AssignNewValue$batch_normalization_7/AssignNewValue2P
&batch_normalization_7/AssignNewValue_1&batch_normalization_7/AssignNewValue_12L
$batch_normalization_8/AssignNewValue$batch_normalization_8/AssignNewValue2P
&batch_normalization_8/AssignNewValue_1&batch_normalization_8/AssignNewValue_12L
$batch_normalization_9/AssignNewValue$batch_normalization_9/AssignNewValue2P
&batch_normalization_9/AssignNewValue_1&batch_normalization_9/AssignNewValue_1:T P
+
_output_shapes
:?????????1(
!
_user_specified_name	input_1: 

_output_shapes
: : 


_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0: (

_output_shapes
:0: -

_output_shapes
:0: 7

_output_shapes
:`: <

_output_shapes
:`

B
&__inference_re_lu_8_layer_call_fn_6686

inputs
identityพ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_re_lu_8_layer_call_and_return_conditional_losses_32102
PartitionedCallk
IdentityIdentityPartitionedCall:output:0*
T0*&
_output_shapes
:`2

Identity"
identityIdentity:output:0*%
_input_shapes
:`:N J
&
_output_shapes
:`
 
_user_specified_nameinputs

B
&__inference_re_lu_2_layer_call_fn_5588

inputs
identityพ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_re_lu_2_layer_call_and_return_conditional_losses_24722
PartitionedCallk
IdentityIdentityPartitionedCall:output:0*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*%
_input_shapes
:0:N J
&
_output_shapes
:0
 
_user_specified_nameinputs

B
&__inference_re_lu_5_layer_call_fn_6210

inputs
identityพ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_re_lu_5_layer_call_and_return_conditional_losses_28912
PartitionedCallk
IdentityIdentityPartitionedCall:output:0*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*%
_input_shapes
:0:N J
&
_output_shapes
:0
 
_user_specified_nameinputs
ฏ
]
A__inference_re_lu_8_layer_call_and_return_conditional_losses_6681

inputs
identityP
Relu6Relu6inputs*
T0*&
_output_shapes
:`2
Relu6f
IdentityIdentityRelu6:activations:0*
T0*&
_output_shapes
:`2

Identity"
identityIdentity:output:0*%
_input_shapes
:`:N J
&
_output_shapes
:`
 
_user_specified_nameinputs
๘

P__inference_batch_normalization_12_layer_call_and_return_conditional_losses_6771

inputs/
+readvariableop_batch_normalization_12_gamma0
,readvariableop_1_batch_normalization_12_betaF
Bfusedbatchnormv3_readvariableop_batch_normalization_12_moving_meanL
Hfusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance
identityขAssignNewValueขAssignNewValue_1
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_12_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_12_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1ม
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_12_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpห
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1๊
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
ื#<2
FusedBatchNormV3ณ
AssignNewValueAssignVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_12_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*U
_classK
IGloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_12/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueษ
AssignNewValue_1AssignVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*[
_classQ
OMloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_12/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1ฆ
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs


O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_6165

inputs-
)readvariableop_batch_normalization_7_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_7_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance0
,fusedbatchnormv3_batch_normalization_7_const
identityขAssignNewValueขAssignNewValue_1
ReadVariableOpReadVariableOp)readvariableop_batch_normalization_7_beta*
_output_shapes
:0*
dtype02
ReadVariableOpภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_7_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputs,fusedbatchnormv3_batch_normalization_7_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
ื#<2
FusedBatchNormV3ฑ
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_7_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_7/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueว
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_7/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1ฆ
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:+???????????????????????????0::::02 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs: 

_output_shapes
:0
ฏ
]
A__inference_re_lu_2_layer_call_and_return_conditional_losses_2472

inputs
identityP
Relu6Relu6inputs*
T0*&
_output_shapes
:02
Relu6f
IdentityIdentityRelu6:activations:0*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*%
_input_shapes
:0:N J
&
_output_shapes
:0
 
_user_specified_nameinputs
ห	

4__inference_batch_normalization_9_layer_call_fn_6382

inputs
batch_normalization_9_gamma
batch_normalization_9_beta%
!batch_normalization_9_moving_mean)
%batch_normalization_9_moving_variance
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_9_gammabatch_normalization_9_beta!batch_normalization_9_moving_mean%batch_normalization_9_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_9_layer_call_and_return_conditional_losses_18622
StatefulPartitionedCallจ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
ฏ
]
A__inference_re_lu_4_layer_call_and_return_conditional_losses_5957

inputs
identityP
Relu6Relu6inputs*
T0*&
_output_shapes
:02
Relu6f
IdentityIdentityRelu6:activations:0*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*%
_input_shapes
:0:N J
&
_output_shapes
:0
 
_user_specified_nameinputs
๐
๖
2__inference_batch_normalization_layer_call_fn_5337

inputs
batch_normalization_beta#
batch_normalization_moving_mean'
#batch_normalization_moving_variance
batch_normalization_const
identityขStatefulPartitionedCallๆ
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_betabatch_normalization_moving_mean#batch_normalization_moving_variancebatch_normalization_const*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_22782
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*7
_input_shapes&
$: :::: 22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
: 
 
_user_specified_nameinputs: 

_output_shapes
: 
?ฎ
0
__inference__wrapped_model_926
input_1;
7functional_1_conv2d_conv2d_readvariableop_conv2d_kernelL
Hfunctional_1_batch_normalization_readvariableop_batch_normalization_betad
`functional_1_batch_normalization_fusedbatchnormv3_readvariableop_batch_normalization_moving_meanj
ffunctional_1_batch_normalization_fusedbatchnormv3_readvariableop_1_batch_normalization_moving_varianceO
Kfunctional_1_batch_normalization_fusedbatchnormv3_batch_normalization_const?
;functional_1_conv2d_1_conv2d_readvariableop_conv2d_1_kernelP
Lfunctional_1_batch_normalization_1_readvariableop_batch_normalization_1_betah
dfunctional_1_batch_normalization_1_fusedbatchnormv3_readvariableop_batch_normalization_1_moving_meann
jfunctional_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_varianceS
Ofunctional_1_batch_normalization_1_fusedbatchnormv3_batch_normalization_1_const\
Xfunctional_1_depthwise_conv2d_depthwise_readvariableop_depthwise_conv2d_depthwise_kernelP
Lfunctional_1_batch_normalization_2_readvariableop_batch_normalization_2_betah
dfunctional_1_batch_normalization_2_fusedbatchnormv3_readvariableop_batch_normalization_2_moving_meann
jfunctional_1_batch_normalization_2_fusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_varianceS
Ofunctional_1_batch_normalization_2_fusedbatchnormv3_batch_normalization_2_const?
;functional_1_conv2d_2_conv2d_readvariableop_conv2d_2_kernelQ
Mfunctional_1_batch_normalization_3_readvariableop_batch_normalization_3_gammaR
Nfunctional_1_batch_normalization_3_readvariableop_1_batch_normalization_3_betah
dfunctional_1_batch_normalization_3_fusedbatchnormv3_readvariableop_batch_normalization_3_moving_meann
jfunctional_1_batch_normalization_3_fusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance?
;functional_1_conv2d_3_conv2d_readvariableop_conv2d_3_kernelP
Lfunctional_1_batch_normalization_4_readvariableop_batch_normalization_4_betah
dfunctional_1_batch_normalization_4_fusedbatchnormv3_readvariableop_batch_normalization_4_moving_meann
jfunctional_1_batch_normalization_4_fusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_varianceS
Ofunctional_1_batch_normalization_4_fusedbatchnormv3_batch_normalization_4_const`
\functional_1_depthwise_conv2d_1_depthwise_readvariableop_depthwise_conv2d_1_depthwise_kernelP
Lfunctional_1_batch_normalization_5_readvariableop_batch_normalization_5_betah
dfunctional_1_batch_normalization_5_fusedbatchnormv3_readvariableop_batch_normalization_5_moving_meann
jfunctional_1_batch_normalization_5_fusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_varianceS
Ofunctional_1_batch_normalization_5_fusedbatchnormv3_batch_normalization_5_const?
;functional_1_conv2d_4_conv2d_readvariableop_conv2d_4_kernelQ
Mfunctional_1_batch_normalization_6_readvariableop_batch_normalization_6_gammaR
Nfunctional_1_batch_normalization_6_readvariableop_1_batch_normalization_6_betah
dfunctional_1_batch_normalization_6_fusedbatchnormv3_readvariableop_batch_normalization_6_moving_meann
jfunctional_1_batch_normalization_6_fusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance?
;functional_1_conv2d_5_conv2d_readvariableop_conv2d_5_kernelP
Lfunctional_1_batch_normalization_7_readvariableop_batch_normalization_7_betah
dfunctional_1_batch_normalization_7_fusedbatchnormv3_readvariableop_batch_normalization_7_moving_meann
jfunctional_1_batch_normalization_7_fusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_varianceS
Ofunctional_1_batch_normalization_7_fusedbatchnormv3_batch_normalization_7_const`
\functional_1_depthwise_conv2d_2_depthwise_readvariableop_depthwise_conv2d_2_depthwise_kernelP
Lfunctional_1_batch_normalization_8_readvariableop_batch_normalization_8_betah
dfunctional_1_batch_normalization_8_fusedbatchnormv3_readvariableop_batch_normalization_8_moving_meann
jfunctional_1_batch_normalization_8_fusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_varianceS
Ofunctional_1_batch_normalization_8_fusedbatchnormv3_batch_normalization_8_const?
;functional_1_conv2d_6_conv2d_readvariableop_conv2d_6_kernelQ
Mfunctional_1_batch_normalization_9_readvariableop_batch_normalization_9_gammaR
Nfunctional_1_batch_normalization_9_readvariableop_1_batch_normalization_9_betah
dfunctional_1_batch_normalization_9_fusedbatchnormv3_readvariableop_batch_normalization_9_moving_meann
jfunctional_1_batch_normalization_9_fusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance?
;functional_1_conv2d_7_conv2d_readvariableop_conv2d_7_kernelR
Nfunctional_1_batch_normalization_10_readvariableop_batch_normalization_10_betaj
ffunctional_1_batch_normalization_10_fusedbatchnormv3_readvariableop_batch_normalization_10_moving_meanp
lfunctional_1_batch_normalization_10_fusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_varianceU
Qfunctional_1_batch_normalization_10_fusedbatchnormv3_batch_normalization_10_const`
\functional_1_depthwise_conv2d_3_depthwise_readvariableop_depthwise_conv2d_3_depthwise_kernelR
Nfunctional_1_batch_normalization_11_readvariableop_batch_normalization_11_betaj
ffunctional_1_batch_normalization_11_fusedbatchnormv3_readvariableop_batch_normalization_11_moving_meanp
lfunctional_1_batch_normalization_11_fusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_varianceU
Qfunctional_1_batch_normalization_11_fusedbatchnormv3_batch_normalization_11_const?
;functional_1_conv2d_8_conv2d_readvariableop_conv2d_8_kernelS
Ofunctional_1_batch_normalization_12_readvariableop_batch_normalization_12_gammaT
Pfunctional_1_batch_normalization_12_readvariableop_1_batch_normalization_12_betaj
ffunctional_1_batch_normalization_12_fusedbatchnormv3_readvariableop_batch_normalization_12_moving_meanp
lfunctional_1_batch_normalization_12_fusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance9
5functional_1_dense_matmul_readvariableop_dense_kernel8
4functional_1_dense_biasadd_readvariableop_dense_bias
identityช
2functional_1/tf_op_layer_ExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :24
2functional_1/tf_op_layer_ExpandDims/ExpandDims/dim๔
.functional_1/tf_op_layer_ExpandDims/ExpandDims
ExpandDimsinput_1;functional_1/tf_op_layer_ExpandDims/ExpandDims/dim:output:0*
T0*
_cloned(*&
_output_shapes
:1(20
.functional_1/tf_op_layer_ExpandDims/ExpandDimsึ
)functional_1/conv2d/Conv2D/ReadVariableOpReadVariableOp7functional_1_conv2d_conv2d_readvariableop_conv2d_kernel*&
_output_shapes
:( *
dtype02+
)functional_1/conv2d/Conv2D/ReadVariableOp
functional_1/conv2d/Conv2DConv2D7functional_1/tf_op_layer_ExpandDims/ExpandDims:output:01functional_1/conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
2
functional_1/conv2d/Conv2D็
/functional_1/batch_normalization/ReadVariableOpReadVariableOpHfunctional_1_batch_normalization_readvariableop_batch_normalization_beta*
_output_shapes
: *
dtype021
/functional_1/batch_normalization/ReadVariableOpก
@functional_1/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp`functional_1_batch_normalization_fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
_output_shapes
: *
dtype02B
@functional_1/batch_normalization/FusedBatchNormV3/ReadVariableOpซ
Bfunctional_1/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpffunctional_1_batch_normalization_fusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
_output_shapes
: *
dtype02D
Bfunctional_1/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ถ
1functional_1/batch_normalization/FusedBatchNormV3FusedBatchNormV3#functional_1/conv2d/Conv2D:output:0Kfunctional_1_batch_normalization_fusedbatchnormv3_batch_normalization_const7functional_1/batch_normalization/ReadVariableOp:value:0Hfunctional_1/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Jfunctional_1/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
is_training( 23
1functional_1/batch_normalization/FusedBatchNormV3ฅ
functional_1/re_lu/Relu6Relu65functional_1/batch_normalization/FusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2
functional_1/re_lu/Relu6?
+functional_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp;functional_1_conv2d_1_conv2d_readvariableop_conv2d_1_kernel*&
_output_shapes
: 0*
dtype02-
+functional_1/conv2d_1/Conv2D/ReadVariableOp?
functional_1/conv2d_1/Conv2DConv2D&functional_1/re_lu/Relu6:activations:03functional_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:0*
paddingSAME*
strides
2
functional_1/conv2d_1/Conv2D๏
1functional_1/batch_normalization_1/ReadVariableOpReadVariableOpLfunctional_1_batch_normalization_1_readvariableop_batch_normalization_1_beta*
_output_shapes
:0*
dtype023
1functional_1/batch_normalization_1/ReadVariableOpฉ
Bfunctional_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpdfunctional_1_batch_normalization_1_fusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
:0*
dtype02D
Bfunctional_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpณ
Dfunctional_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpjfunctional_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
_output_shapes
:0*
dtype02F
Dfunctional_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ฦ
3functional_1/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3%functional_1/conv2d_1/Conv2D:output:0Ofunctional_1_batch_normalization_1_fusedbatchnormv3_batch_normalization_1_const9functional_1/batch_normalization_1/ReadVariableOp:value:0Jfunctional_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Lfunctional_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:0:0:0:0:0:*
epsilon%o:*
is_training( 25
3functional_1/batch_normalization_1/FusedBatchNormV3ซ
functional_1/re_lu_1/Relu6Relu67functional_1/batch_normalization_1/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:02
functional_1/re_lu_1/Relu6
6functional_1/depthwise_conv2d/depthwise/ReadVariableOpReadVariableOpXfunctional_1_depthwise_conv2d_depthwise_readvariableop_depthwise_conv2d_depthwise_kernel*&
_output_shapes
:0*
dtype028
6functional_1/depthwise_conv2d/depthwise/ReadVariableOpท
-functional_1/depthwise_conv2d/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      0      2/
-functional_1/depthwise_conv2d/depthwise/Shapeฟ
5functional_1/depthwise_conv2d/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      27
5functional_1/depthwise_conv2d/depthwise/dilation_rateฎ
'functional_1/depthwise_conv2d/depthwiseDepthwiseConv2dNative(functional_1/re_lu_1/Relu6:activations:0>functional_1/depthwise_conv2d/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:0*
paddingSAME*
strides
2)
'functional_1/depthwise_conv2d/depthwise๏
1functional_1/batch_normalization_2/ReadVariableOpReadVariableOpLfunctional_1_batch_normalization_2_readvariableop_batch_normalization_2_beta*
_output_shapes
:0*
dtype023
1functional_1/batch_normalization_2/ReadVariableOpฉ
Bfunctional_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpdfunctional_1_batch_normalization_2_fusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
:0*
dtype02D
Bfunctional_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOpณ
Dfunctional_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpjfunctional_1_batch_normalization_2_fusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
_output_shapes
:0*
dtype02F
Dfunctional_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ั
3functional_1/batch_normalization_2/FusedBatchNormV3FusedBatchNormV30functional_1/depthwise_conv2d/depthwise:output:0Ofunctional_1_batch_normalization_2_fusedbatchnormv3_batch_normalization_2_const9functional_1/batch_normalization_2/ReadVariableOp:value:0Jfunctional_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Lfunctional_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:0:0:0:0:0:*
epsilon%o:*
is_training( 25
3functional_1/batch_normalization_2/FusedBatchNormV3ซ
functional_1/re_lu_2/Relu6Relu67functional_1/batch_normalization_2/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:02
functional_1/re_lu_2/Relu6?
+functional_1/conv2d_2/Conv2D/ReadVariableOpReadVariableOp;functional_1_conv2d_2_conv2d_readvariableop_conv2d_2_kernel*&
_output_shapes
:0 *
dtype02-
+functional_1/conv2d_2/Conv2D/ReadVariableOp?
functional_1/conv2d_2/Conv2DConv2D(functional_1/re_lu_2/Relu6:activations:03functional_1/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingSAME*
strides
2
functional_1/conv2d_2/Conv2D๐
1functional_1/batch_normalization_3/ReadVariableOpReadVariableOpMfunctional_1_batch_normalization_3_readvariableop_batch_normalization_3_gamma*
_output_shapes
: *
dtype023
1functional_1/batch_normalization_3/ReadVariableOp๕
3functional_1/batch_normalization_3/ReadVariableOp_1ReadVariableOpNfunctional_1_batch_normalization_3_readvariableop_1_batch_normalization_3_beta*
_output_shapes
: *
dtype025
3functional_1/batch_normalization_3/ReadVariableOp_1ฉ
Bfunctional_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpdfunctional_1_batch_normalization_3_fusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
_output_shapes
: *
dtype02D
Bfunctional_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpณ
Dfunctional_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpjfunctional_1_batch_normalization_3_fusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
_output_shapes
: *
dtype02F
Dfunctional_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ฒ
3functional_1/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3%functional_1/conv2d_2/Conv2D:output:09functional_1/batch_normalization_3/ReadVariableOp:value:0;functional_1/batch_normalization_3/ReadVariableOp_1:value:0Jfunctional_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Lfunctional_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
is_training( 25
3functional_1/batch_normalization_3/FusedBatchNormV3ว
functional_1/add/addAddV2&functional_1/re_lu/Relu6:activations:07functional_1/batch_normalization_3/FusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2
functional_1/add/add?
+functional_1/conv2d_3/Conv2D/ReadVariableOpReadVariableOp;functional_1_conv2d_3_conv2d_readvariableop_conv2d_3_kernel*&
_output_shapes
: 0*
dtype02-
+functional_1/conv2d_3/Conv2D/ReadVariableOp๎
functional_1/conv2d_3/Conv2DConv2Dfunctional_1/add/add:z:03functional_1/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:0*
paddingSAME*
strides
2
functional_1/conv2d_3/Conv2D๏
1functional_1/batch_normalization_4/ReadVariableOpReadVariableOpLfunctional_1_batch_normalization_4_readvariableop_batch_normalization_4_beta*
_output_shapes
:0*
dtype023
1functional_1/batch_normalization_4/ReadVariableOpฉ
Bfunctional_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpdfunctional_1_batch_normalization_4_fusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
_output_shapes
:0*
dtype02D
Bfunctional_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOpณ
Dfunctional_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpjfunctional_1_batch_normalization_4_fusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
_output_shapes
:0*
dtype02F
Dfunctional_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ฦ
3functional_1/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3%functional_1/conv2d_3/Conv2D:output:0Ofunctional_1_batch_normalization_4_fusedbatchnormv3_batch_normalization_4_const9functional_1/batch_normalization_4/ReadVariableOp:value:0Jfunctional_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Lfunctional_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:0:0:0:0:0:*
epsilon%o:*
is_training( 25
3functional_1/batch_normalization_4/FusedBatchNormV3ซ
functional_1/re_lu_3/Relu6Relu67functional_1/batch_normalization_4/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:02
functional_1/re_lu_3/Relu6
8functional_1/depthwise_conv2d_1/depthwise/ReadVariableOpReadVariableOp\functional_1_depthwise_conv2d_1_depthwise_readvariableop_depthwise_conv2d_1_depthwise_kernel*&
_output_shapes
:0*
dtype02:
8functional_1/depthwise_conv2d_1/depthwise/ReadVariableOpป
/functional_1/depthwise_conv2d_1/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      0      21
/functional_1/depthwise_conv2d_1/depthwise/Shapeร
7functional_1/depthwise_conv2d_1/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      29
7functional_1/depthwise_conv2d_1/depthwise/dilation_rateด
)functional_1/depthwise_conv2d_1/depthwiseDepthwiseConv2dNative(functional_1/re_lu_3/Relu6:activations:0@functional_1/depthwise_conv2d_1/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:0*
paddingSAME*
strides
2+
)functional_1/depthwise_conv2d_1/depthwise๏
1functional_1/batch_normalization_5/ReadVariableOpReadVariableOpLfunctional_1_batch_normalization_5_readvariableop_batch_normalization_5_beta*
_output_shapes
:0*
dtype023
1functional_1/batch_normalization_5/ReadVariableOpฉ
Bfunctional_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpdfunctional_1_batch_normalization_5_fusedbatchnormv3_readvariableop_batch_normalization_5_moving_mean*
_output_shapes
:0*
dtype02D
Bfunctional_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpณ
Dfunctional_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpjfunctional_1_batch_normalization_5_fusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance*
_output_shapes
:0*
dtype02F
Dfunctional_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ำ
3functional_1/batch_normalization_5/FusedBatchNormV3FusedBatchNormV32functional_1/depthwise_conv2d_1/depthwise:output:0Ofunctional_1_batch_normalization_5_fusedbatchnormv3_batch_normalization_5_const9functional_1/batch_normalization_5/ReadVariableOp:value:0Jfunctional_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Lfunctional_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:0:0:0:0:0:*
epsilon%o:*
is_training( 25
3functional_1/batch_normalization_5/FusedBatchNormV3ซ
functional_1/re_lu_4/Relu6Relu67functional_1/batch_normalization_5/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:02
functional_1/re_lu_4/Relu6?
+functional_1/conv2d_4/Conv2D/ReadVariableOpReadVariableOp;functional_1_conv2d_4_conv2d_readvariableop_conv2d_4_kernel*&
_output_shapes
:0 *
dtype02-
+functional_1/conv2d_4/Conv2D/ReadVariableOp?
functional_1/conv2d_4/Conv2DConv2D(functional_1/re_lu_4/Relu6:activations:03functional_1/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingSAME*
strides
2
functional_1/conv2d_4/Conv2D๐
1functional_1/batch_normalization_6/ReadVariableOpReadVariableOpMfunctional_1_batch_normalization_6_readvariableop_batch_normalization_6_gamma*
_output_shapes
: *
dtype023
1functional_1/batch_normalization_6/ReadVariableOp๕
3functional_1/batch_normalization_6/ReadVariableOp_1ReadVariableOpNfunctional_1_batch_normalization_6_readvariableop_1_batch_normalization_6_beta*
_output_shapes
: *
dtype025
3functional_1/batch_normalization_6/ReadVariableOp_1ฉ
Bfunctional_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpdfunctional_1_batch_normalization_6_fusedbatchnormv3_readvariableop_batch_normalization_6_moving_mean*
_output_shapes
: *
dtype02D
Bfunctional_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpณ
Dfunctional_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpjfunctional_1_batch_normalization_6_fusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance*
_output_shapes
: *
dtype02F
Dfunctional_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ฒ
3functional_1/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3%functional_1/conv2d_4/Conv2D:output:09functional_1/batch_normalization_6/ReadVariableOp:value:0;functional_1/batch_normalization_6/ReadVariableOp_1:value:0Jfunctional_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Lfunctional_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
is_training( 25
3functional_1/batch_normalization_6/FusedBatchNormV3?
+functional_1/conv2d_5/Conv2D/ReadVariableOpReadVariableOp;functional_1_conv2d_5_conv2d_readvariableop_conv2d_5_kernel*&
_output_shapes
: 0*
dtype02-
+functional_1/conv2d_5/Conv2D/ReadVariableOp
functional_1/conv2d_5/Conv2DConv2D7functional_1/batch_normalization_6/FusedBatchNormV3:y:03functional_1/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:0*
paddingSAME*
strides
2
functional_1/conv2d_5/Conv2D๏
1functional_1/batch_normalization_7/ReadVariableOpReadVariableOpLfunctional_1_batch_normalization_7_readvariableop_batch_normalization_7_beta*
_output_shapes
:0*
dtype023
1functional_1/batch_normalization_7/ReadVariableOpฉ
Bfunctional_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpdfunctional_1_batch_normalization_7_fusedbatchnormv3_readvariableop_batch_normalization_7_moving_mean*
_output_shapes
:0*
dtype02D
Bfunctional_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOpณ
Dfunctional_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpjfunctional_1_batch_normalization_7_fusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance*
_output_shapes
:0*
dtype02F
Dfunctional_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ฦ
3functional_1/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3%functional_1/conv2d_5/Conv2D:output:0Ofunctional_1_batch_normalization_7_fusedbatchnormv3_batch_normalization_7_const9functional_1/batch_normalization_7/ReadVariableOp:value:0Jfunctional_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Lfunctional_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:0:0:0:0:0:*
epsilon%o:*
is_training( 25
3functional_1/batch_normalization_7/FusedBatchNormV3ซ
functional_1/re_lu_5/Relu6Relu67functional_1/batch_normalization_7/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:02
functional_1/re_lu_5/Relu6
8functional_1/depthwise_conv2d_2/depthwise/ReadVariableOpReadVariableOp\functional_1_depthwise_conv2d_2_depthwise_readvariableop_depthwise_conv2d_2_depthwise_kernel*&
_output_shapes
:0*
dtype02:
8functional_1/depthwise_conv2d_2/depthwise/ReadVariableOpป
/functional_1/depthwise_conv2d_2/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      0      21
/functional_1/depthwise_conv2d_2/depthwise/Shapeร
7functional_1/depthwise_conv2d_2/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      29
7functional_1/depthwise_conv2d_2/depthwise/dilation_rateด
)functional_1/depthwise_conv2d_2/depthwiseDepthwiseConv2dNative(functional_1/re_lu_5/Relu6:activations:0@functional_1/depthwise_conv2d_2/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:0*
paddingSAME*
strides
2+
)functional_1/depthwise_conv2d_2/depthwise๏
1functional_1/batch_normalization_8/ReadVariableOpReadVariableOpLfunctional_1_batch_normalization_8_readvariableop_batch_normalization_8_beta*
_output_shapes
:0*
dtype023
1functional_1/batch_normalization_8/ReadVariableOpฉ
Bfunctional_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpdfunctional_1_batch_normalization_8_fusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean*
_output_shapes
:0*
dtype02D
Bfunctional_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOpณ
Dfunctional_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpjfunctional_1_batch_normalization_8_fusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance*
_output_shapes
:0*
dtype02F
Dfunctional_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ำ
3functional_1/batch_normalization_8/FusedBatchNormV3FusedBatchNormV32functional_1/depthwise_conv2d_2/depthwise:output:0Ofunctional_1_batch_normalization_8_fusedbatchnormv3_batch_normalization_8_const9functional_1/batch_normalization_8/ReadVariableOp:value:0Jfunctional_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Lfunctional_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:0:0:0:0:0:*
epsilon%o:*
is_training( 25
3functional_1/batch_normalization_8/FusedBatchNormV3ซ
functional_1/re_lu_6/Relu6Relu67functional_1/batch_normalization_8/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:02
functional_1/re_lu_6/Relu6?
+functional_1/conv2d_6/Conv2D/ReadVariableOpReadVariableOp;functional_1_conv2d_6_conv2d_readvariableop_conv2d_6_kernel*&
_output_shapes
:0@*
dtype02-
+functional_1/conv2d_6/Conv2D/ReadVariableOp?
functional_1/conv2d_6/Conv2DConv2D(functional_1/re_lu_6/Relu6:activations:03functional_1/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingSAME*
strides
2
functional_1/conv2d_6/Conv2D๐
1functional_1/batch_normalization_9/ReadVariableOpReadVariableOpMfunctional_1_batch_normalization_9_readvariableop_batch_normalization_9_gamma*
_output_shapes
:@*
dtype023
1functional_1/batch_normalization_9/ReadVariableOp๕
3functional_1/batch_normalization_9/ReadVariableOp_1ReadVariableOpNfunctional_1_batch_normalization_9_readvariableop_1_batch_normalization_9_beta*
_output_shapes
:@*
dtype025
3functional_1/batch_normalization_9/ReadVariableOp_1ฉ
Bfunctional_1/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpdfunctional_1_batch_normalization_9_fusedbatchnormv3_readvariableop_batch_normalization_9_moving_mean*
_output_shapes
:@*
dtype02D
Bfunctional_1/batch_normalization_9/FusedBatchNormV3/ReadVariableOpณ
Dfunctional_1/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpjfunctional_1_batch_normalization_9_fusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance*
_output_shapes
:@*
dtype02F
Dfunctional_1/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ฒ
3functional_1/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3%functional_1/conv2d_6/Conv2D:output:09functional_1/batch_normalization_9/ReadVariableOp:value:0;functional_1/batch_normalization_9/ReadVariableOp_1:value:0Jfunctional_1/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Lfunctional_1/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:@:@:@:@:@:*
epsilon%o:*
is_training( 25
3functional_1/batch_normalization_9/FusedBatchNormV3?
+functional_1/conv2d_7/Conv2D/ReadVariableOpReadVariableOp;functional_1_conv2d_7_conv2d_readvariableop_conv2d_7_kernel*&
_output_shapes
:@`*
dtype02-
+functional_1/conv2d_7/Conv2D/ReadVariableOp
functional_1/conv2d_7/Conv2DConv2D7functional_1/batch_normalization_9/FusedBatchNormV3:y:03functional_1/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:`*
paddingSAME*
strides
2
functional_1/conv2d_7/Conv2D๓
2functional_1/batch_normalization_10/ReadVariableOpReadVariableOpNfunctional_1_batch_normalization_10_readvariableop_batch_normalization_10_beta*
_output_shapes
:`*
dtype024
2functional_1/batch_normalization_10/ReadVariableOpญ
Cfunctional_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOpffunctional_1_batch_normalization_10_fusedbatchnormv3_readvariableop_batch_normalization_10_moving_mean*
_output_shapes
:`*
dtype02E
Cfunctional_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOpท
Efunctional_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOplfunctional_1_batch_normalization_10_fusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance*
_output_shapes
:`*
dtype02G
Efunctional_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1อ
4functional_1/batch_normalization_10/FusedBatchNormV3FusedBatchNormV3%functional_1/conv2d_7/Conv2D:output:0Qfunctional_1_batch_normalization_10_fusedbatchnormv3_batch_normalization_10_const:functional_1/batch_normalization_10/ReadVariableOp:value:0Kfunctional_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0Mfunctional_1/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:`:`:`:`:`:*
epsilon%o:*
is_training( 26
4functional_1/batch_normalization_10/FusedBatchNormV3ฌ
functional_1/re_lu_7/Relu6Relu68functional_1/batch_normalization_10/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:`2
functional_1/re_lu_7/Relu6
8functional_1/depthwise_conv2d_3/depthwise/ReadVariableOpReadVariableOp\functional_1_depthwise_conv2d_3_depthwise_readvariableop_depthwise_conv2d_3_depthwise_kernel*&
_output_shapes
:`*
dtype02:
8functional_1/depthwise_conv2d_3/depthwise/ReadVariableOpป
/functional_1/depthwise_conv2d_3/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      `      21
/functional_1/depthwise_conv2d_3/depthwise/Shapeร
7functional_1/depthwise_conv2d_3/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      29
7functional_1/depthwise_conv2d_3/depthwise/dilation_rateด
)functional_1/depthwise_conv2d_3/depthwiseDepthwiseConv2dNative(functional_1/re_lu_7/Relu6:activations:0@functional_1/depthwise_conv2d_3/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:`*
paddingSAME*
strides
2+
)functional_1/depthwise_conv2d_3/depthwise๓
2functional_1/batch_normalization_11/ReadVariableOpReadVariableOpNfunctional_1_batch_normalization_11_readvariableop_batch_normalization_11_beta*
_output_shapes
:`*
dtype024
2functional_1/batch_normalization_11/ReadVariableOpญ
Cfunctional_1/batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOpffunctional_1_batch_normalization_11_fusedbatchnormv3_readvariableop_batch_normalization_11_moving_mean*
_output_shapes
:`*
dtype02E
Cfunctional_1/batch_normalization_11/FusedBatchNormV3/ReadVariableOpท
Efunctional_1/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOplfunctional_1_batch_normalization_11_fusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance*
_output_shapes
:`*
dtype02G
Efunctional_1/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ฺ
4functional_1/batch_normalization_11/FusedBatchNormV3FusedBatchNormV32functional_1/depthwise_conv2d_3/depthwise:output:0Qfunctional_1_batch_normalization_11_fusedbatchnormv3_batch_normalization_11_const:functional_1/batch_normalization_11/ReadVariableOp:value:0Kfunctional_1/batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0Mfunctional_1/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:`:`:`:`:`:*
epsilon%o:*
is_training( 26
4functional_1/batch_normalization_11/FusedBatchNormV3ฌ
functional_1/re_lu_8/Relu6Relu68functional_1/batch_normalization_11/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:`2
functional_1/re_lu_8/Relu6?
+functional_1/conv2d_8/Conv2D/ReadVariableOpReadVariableOp;functional_1_conv2d_8_conv2d_readvariableop_conv2d_8_kernel*&
_output_shapes
:`@*
dtype02-
+functional_1/conv2d_8/Conv2D/ReadVariableOp?
functional_1/conv2d_8/Conv2DConv2D(functional_1/re_lu_8/Relu6:activations:03functional_1/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingSAME*
strides
2
functional_1/conv2d_8/Conv2D๔
2functional_1/batch_normalization_12/ReadVariableOpReadVariableOpOfunctional_1_batch_normalization_12_readvariableop_batch_normalization_12_gamma*
_output_shapes
:@*
dtype024
2functional_1/batch_normalization_12/ReadVariableOp๙
4functional_1/batch_normalization_12/ReadVariableOp_1ReadVariableOpPfunctional_1_batch_normalization_12_readvariableop_1_batch_normalization_12_beta*
_output_shapes
:@*
dtype026
4functional_1/batch_normalization_12/ReadVariableOp_1ญ
Cfunctional_1/batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOpffunctional_1_batch_normalization_12_fusedbatchnormv3_readvariableop_batch_normalization_12_moving_mean*
_output_shapes
:@*
dtype02E
Cfunctional_1/batch_normalization_12/FusedBatchNormV3/ReadVariableOpท
Efunctional_1/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOplfunctional_1_batch_normalization_12_fusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance*
_output_shapes
:@*
dtype02G
Efunctional_1/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ธ
4functional_1/batch_normalization_12/FusedBatchNormV3FusedBatchNormV3%functional_1/conv2d_8/Conv2D:output:0:functional_1/batch_normalization_12/ReadVariableOp:value:0<functional_1/batch_normalization_12/ReadVariableOp_1:value:0Kfunctional_1/batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0Mfunctional_1/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:@:@:@:@:@:*
epsilon%o:*
is_training( 26
4functional_1/batch_normalization_12/FusedBatchNormV3?
functional_1/add_1/addAddV27functional_1/batch_normalization_9/FusedBatchNormV3:y:08functional_1/batch_normalization_12/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:@2
functional_1/add_1/addอ
<functional_1/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2>
<functional_1/global_average_pooling2d/Mean/reduction_indices์
*functional_1/global_average_pooling2d/MeanMeanfunctional_1/add_1/add:z:0Efunctional_1/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*
_output_shapes

:@2,
*functional_1/global_average_pooling2d/Meanจ
functional_1/dropout/IdentityIdentity3functional_1/global_average_pooling2d/Mean:output:0*
T0*
_output_shapes

:@2
functional_1/dropout/Identityส
(functional_1/dense/MatMul/ReadVariableOpReadVariableOp5functional_1_dense_matmul_readvariableop_dense_kernel*
_output_shapes

:@*
dtype02*
(functional_1/dense/MatMul/ReadVariableOpร
functional_1/dense/MatMulMatMul&functional_1/dropout/Identity:output:00functional_1/dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/dense/MatMulว
)functional_1/dense/BiasAdd/ReadVariableOpReadVariableOp4functional_1_dense_biasadd_readvariableop_dense_bias*
_output_shapes
:*
dtype02+
)functional_1/dense/BiasAdd/ReadVariableOpฤ
functional_1/dense/BiasAddBiasAdd#functional_1/dense/MatMul:product:01functional_1/dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/dense/BiasAddn
IdentityIdentity#functional_1/dense/BiasAdd:output:0*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*ม
_input_shapesฏ
ฌ:1(::::: :::::0:::::0::::::::::0:::::0::::::::::0:::::0::::::::::`:::::`::::::::T P
+
_output_shapes
:?????????1(
!
_user_specified_name	input_1: 

_output_shapes
: : 


_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0: (

_output_shapes
:0: -

_output_shapes
:0: 7

_output_shapes
:`: <

_output_shapes
:`

B
&__inference_re_lu_6_layer_call_fn_6324

inputs
identityพ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_re_lu_6_layer_call_and_return_conditional_losses_29692
PartitionedCallk
IdentityIdentityPartitionedCall:output:0*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*%
_input_shapes
:0:N J
&
_output_shapes
:0
 
_user_specified_nameinputs
?

O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_2514

inputs.
*readvariableop_batch_normalization_3_gamma/
+readvariableop_1_batch_normalization_3_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_3_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance
identityขAssignNewValueขAssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_3_gamma*
_output_shapes
: *
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_3_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1ภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ฯ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
exponential_avg_factor%
ื#<2
FusedBatchNormV3ฑ
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_3_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_3/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueว
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_3/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:N J
&
_output_shapes
: 
 
_user_specified_nameinputs
	
ญ
L__inference_depthwise_conv2d_1_layer_call_and_return_conditional_losses_1418

inputs@
<depthwise_readvariableop_depthwise_conv2d_1_depthwise_kernel
identityน
depthwise/ReadVariableOpReadVariableOp<depthwise_readvariableop_depthwise_conv2d_1_depthwise_kernel*&
_output_shapes
:0*
dtype02
depthwise/ReadVariableOp{
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      0      2
depthwise/Shape
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
depthwise/dilation_rateอ
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????0*
paddingSAME*
strides
2
	depthwise
IdentityIdentitydepthwise:output:0*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+???????????????????????????0::i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
	

5__inference_batch_normalization_11_layer_call_fn_6624

inputs
batch_normalization_11_beta&
"batch_normalization_11_moving_mean*
&batch_normalization_11_moving_variance 
batch_normalization_11_const
identityขStatefulPartitionedCall๕
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_11_beta"batch_normalization_11_moving_mean&batch_normalization_11_moving_variancebatch_normalization_11_const*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:`*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_31812
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:`2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:`::::`22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:`
 
_user_specified_nameinputs: 

_output_shapes
:`
	

4__inference_batch_normalization_7_layer_call_fn_6148

inputs
batch_normalization_7_beta%
!batch_normalization_7_moving_mean)
%batch_normalization_7_moving_variance
batch_normalization_7_const
identityขStatefulPartitionedCall๐
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_7_beta!batch_normalization_7_moving_mean%batch_normalization_7_moving_variancebatch_normalization_7_const*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_28622
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:0::::022
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:0
 
_user_specified_nameinputs: 

_output_shapes
:0

ใ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2699

inputs-
)readvariableop_batch_normalization_5_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_5_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance0
,fusedbatchnormv3_batch_normalization_5_const
identity
ReadVariableOpReadVariableOp)readvariableop_batch_normalization_5_beta*
_output_shapes
:0*
dtype02
ReadVariableOpภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_5_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ี
FusedBatchNormV3FusedBatchNormV3inputs,fusedbatchnormv3_batch_normalization_5_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:0:0:0:0:0:*
epsilon%o:*
is_training( 2
FusedBatchNormV3g
IdentityIdentityFusedBatchNormV3:y:0*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:0::::0:N J
&
_output_shapes
:0
 
_user_specified_nameinputs: 

_output_shapes
:0
๎
๖
2__inference_batch_normalization_layer_call_fn_5328

inputs
batch_normalization_beta#
batch_normalization_moving_mean'
#batch_normalization_moving_variance
batch_normalization_const
identityขStatefulPartitionedCallไ
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_betabatch_normalization_moving_mean#batch_normalization_moving_variancebatch_normalization_const*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_22612
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*7
_input_shapes&
$: :::: 22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
: 
 
_user_specified_nameinputs: 

_output_shapes
: 
?

P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_6641

inputs.
*readvariableop_batch_normalization_11_betaF
Bfusedbatchnormv3_readvariableop_batch_normalization_11_moving_meanL
Hfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance1
-fusedbatchnormv3_batch_normalization_11_const
identityขAssignNewValueขAssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_11_beta*
_output_shapes
:`*
dtype02
ReadVariableOpม
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_11_moving_mean*
_output_shapes
:`*
dtype02!
FusedBatchNormV3/ReadVariableOpห
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance*
_output_shapes
:`*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputs-fusedbatchnormv3_batch_normalization_11_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????`:`:`:`:`:*
epsilon%o:*
exponential_avg_factor%
ื#<2
FusedBatchNormV3ณ
AssignNewValueAssignVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_11_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*U
_classK
IGloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_11/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueษ
AssignNewValue_1AssignVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*[
_classQ
OMloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_11/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1ฆ
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+???????????????????????????`2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:+???????????????????????????`::::`2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+???????????????????????????`
 
_user_specified_nameinputs: 

_output_shapes
:`
	

4__inference_batch_normalization_4_layer_call_fn_5777

inputs
batch_normalization_4_beta%
!batch_normalization_4_moving_mean)
%batch_normalization_4_moving_variance
batch_normalization_4_const
identityขStatefulPartitionedCall๎
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_4_beta!batch_normalization_4_moving_mean%batch_normalization_4_moving_variancebatch_normalization_4_const*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_26042
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:0::::022
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:0
 
_user_specified_nameinputs: 

_output_shapes
:0
ธ
n
R__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_2206

inputs
identity
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
๋

O__inference_batch_normalization_9_layer_call_and_return_conditional_losses_6355

inputs.
*readvariableop_batch_normalization_9_gamma/
+readvariableop_1_batch_normalization_9_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_9_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance
identityขAssignNewValueขAssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_9_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_9_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1ภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_9_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1๊
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
ื#<2
FusedBatchNormV3ฑ
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_9_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_9/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueว
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_9/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1ฆ
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
฿

4__inference_batch_normalization_9_layer_call_fn_6436

inputs
batch_normalization_9_gamma
batch_normalization_9_beta%
!batch_normalization_9_moving_mean)
%batch_normalization_9_moving_variance
identityขStatefulPartitionedCall๏
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_9_gammabatch_normalization_9_beta!batch_normalization_9_moving_mean%batch_normalization_9_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_9_layer_call_and_return_conditional_losses_30112
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*5
_input_shapes$
":@::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:@
 
_user_specified_nameinputs
ๅ
ู
M__inference_batch_normalization_layer_call_and_return_conditional_losses_1007

inputs+
'readvariableop_batch_normalization_betaC
?fusedbatchnormv3_readvariableop_batch_normalization_moving_meanI
Efusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance.
*fusedbatchnormv3_batch_normalization_const
identity
ReadVariableOpReadVariableOp'readvariableop_batch_normalization_beta*
_output_shapes
: *
dtype02
ReadVariableOpพ
FusedBatchNormV3/ReadVariableOpReadVariableOp?fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpศ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpEfusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1๎
FusedBatchNormV3FusedBatchNormV3inputs*fusedbatchnormv3_batch_normalization_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:+??????????????????????????? :::: :i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs: 

_output_shapes
: 
ฏ
]
A__inference_re_lu_7_layer_call_and_return_conditional_losses_6567

inputs
identityP
Relu6Relu6inputs*
T0*&
_output_shapes
:`2
Relu6f
IdentityIdentityRelu6:activations:0*
T0*&
_output_shapes
:`2

Identity"
identityIdentity:output:0*%
_input_shapes
:`:N J
&
_output_shapes
:`
 
_user_specified_nameinputs
๚	

5__inference_batch_normalization_11_layer_call_fn_6676

inputs
batch_normalization_11_beta&
"batch_normalization_11_moving_mean*
&batch_normalization_11_moving_variance 
batch_normalization_11_const
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_11_beta"batch_normalization_11_moving_mean&batch_normalization_11_moving_variancebatch_normalization_11_const*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????`*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_20912
StatefulPartitionedCallจ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????`2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:+???????????????????????????`::::`22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????`
 
_user_specified_nameinputs: 

_output_shapes
:`
	

4__inference_batch_normalization_1_layer_call_fn_5464

inputs
batch_normalization_1_beta%
!batch_normalization_1_moving_mean)
%batch_normalization_1_moving_variance
batch_normalization_1_const
identityขStatefulPartitionedCall๐
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_1_beta!batch_normalization_1_moving_mean%batch_normalization_1_moving_variancebatch_normalization_1_const*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_23652
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:0::::022
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:0
 
_user_specified_nameinputs: 

_output_shapes
:0
้

5__inference_batch_normalization_12_layer_call_fn_6744

inputs 
batch_normalization_12_gamma
batch_normalization_12_beta&
"batch_normalization_12_moving_mean*
&batch_normalization_12_moving_variance
identityขStatefulPartitionedCall๔
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_12_gammabatch_normalization_12_beta"batch_normalization_12_moving_mean&batch_normalization_12_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_12_layer_call_and_return_conditional_losses_32522
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*5
_input_shapes$
":@::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:@
 
_user_specified_nameinputs
	
ฉ
J__inference_depthwise_conv2d_layer_call_and_return_conditional_losses_1112

inputs>
:depthwise_readvariableop_depthwise_conv2d_depthwise_kernel
identityท
depthwise/ReadVariableOpReadVariableOp:depthwise_readvariableop_depthwise_conv2d_depthwise_kernel*&
_output_shapes
:0*
dtype02
depthwise/ReadVariableOp{
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      0      2
depthwise/Shape
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
depthwise/dilation_rateอ
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????0*
paddingSAME*
strides
2
	depthwise
IdentityIdentitydepthwise:output:0*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+???????????????????????????0::i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
แ

4__inference_batch_normalization_9_layer_call_fn_6445

inputs
batch_normalization_9_gamma
batch_normalization_9_beta%
!batch_normalization_9_moving_mean)
%batch_normalization_9_moving_variance
identityขStatefulPartitionedCall๑
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_9_gammabatch_normalization_9_beta!batch_normalization_9_moving_mean%batch_normalization_9_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_9_layer_call_and_return_conditional_losses_30292
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*5
_input_shapes$
":@::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:@
 
_user_specified_nameinputs
โ
ใ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2788

inputs.
*readvariableop_batch_normalization_6_gamma/
+readvariableop_1_batch_normalization_6_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_6_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_6_gamma*
_output_shapes
: *
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_6_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1ภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_6_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ม
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3g
IdentityIdentityFusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": :::::N J
&
_output_shapes
: 
 
_user_specified_nameinputs
๐	

4__inference_batch_normalization_4_layer_call_fn_5838

inputs
batch_normalization_4_beta%
!batch_normalization_4_moving_mean)
%batch_normalization_4_moving_variance
batch_normalization_4_const
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_4_beta!batch_normalization_4_moving_mean%batch_normalization_4_moving_variancebatch_normalization_4_const*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????0*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_13892
StatefulPartitionedCallจ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:+???????????????????????????0::::022
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs: 

_output_shapes
:0

?
M__inference_batch_normalization_layer_call_and_return_conditional_losses_2261

inputs+
'readvariableop_batch_normalization_betaC
?fusedbatchnormv3_readvariableop_batch_normalization_moving_meanI
Efusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance.
*fusedbatchnormv3_batch_normalization_const
identityขAssignNewValueขAssignNewValue_1
ReadVariableOpReadVariableOp'readvariableop_batch_normalization_beta*
_output_shapes
: *
dtype02
ReadVariableOpพ
FusedBatchNormV3/ReadVariableOpReadVariableOp?fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpศ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpEfusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1แ
FusedBatchNormV3FusedBatchNormV3inputs*fusedbatchnormv3_batch_normalization_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
exponential_avg_factor%
ื#<2
FusedBatchNormV3ญ
AssignNewValueAssignVariableOp?fusedbatchnormv3_readvariableop_batch_normalization_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueร
AssignNewValue_1AssignVariableOpEfusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*X
_classN
LJloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*7
_input_shapes&
$: :::: 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:N J
&
_output_shapes
: 
 
_user_specified_nameinputs: 

_output_shapes
: 
ฯ
ใ
O__inference_batch_normalization_9_layer_call_and_return_conditional_losses_6373

inputs.
*readvariableop_batch_normalization_9_gamma/
+readvariableop_1_batch_normalization_9_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_9_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_9_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_9_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1ภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_9_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@:::::i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
๘
ู
M__inference_batch_normalization_layer_call_and_return_conditional_losses_5319

inputs+
'readvariableop_batch_normalization_betaC
?fusedbatchnormv3_readvariableop_batch_normalization_moving_meanI
Efusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance.
*fusedbatchnormv3_batch_normalization_const
identity
ReadVariableOpReadVariableOp'readvariableop_batch_normalization_beta*
_output_shapes
: *
dtype02
ReadVariableOpพ
FusedBatchNormV3/ReadVariableOpReadVariableOp?fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpศ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpEfusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ำ
FusedBatchNormV3FusedBatchNormV3inputs*fusedbatchnormv3_batch_normalization_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3g
IdentityIdentityFusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*7
_input_shapes&
$: :::: :N J
&
_output_shapes
: 
 
_user_specified_nameinputs: 

_output_shapes
: 
า
l
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_2219

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*
_cloned(*&
_output_shapes
:1(2

ExpandDimsf
IdentityIdentityExpandDims:output:0*
T0*&
_output_shapes
:1(2

Identity"
identityIdentity:output:0*!
_input_shapes
:1(:J F
"
_output_shapes
:1(
 
_user_specified_nameinputs
	
ฉ
J__inference_depthwise_conv2d_layer_call_and_return_conditional_losses_1124

inputs>
:depthwise_readvariableop_depthwise_conv2d_depthwise_kernel
identityท
depthwise/ReadVariableOpReadVariableOp:depthwise_readvariableop_depthwise_conv2d_depthwise_kernel*&
_output_shapes
:0*
dtype02
depthwise/ReadVariableOp{
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      0      2
depthwise/Shape
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
depthwise/dilation_rateอ
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????0*
paddingSAME*
strides
2
	depthwise
IdentityIdentitydepthwise:output:0*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+???????????????????????????0::i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs

ใ
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_6244

inputs-
)readvariableop_batch_normalization_8_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_8_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance0
,fusedbatchnormv3_batch_normalization_8_const
identity
ReadVariableOpReadVariableOp)readvariableop_batch_normalization_8_beta*
_output_shapes
:0*
dtype02
ReadVariableOpภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ี
FusedBatchNormV3FusedBatchNormV3inputs,fusedbatchnormv3_batch_normalization_8_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:0:0:0:0:0:*
epsilon%o:*
is_training( 2
FusedBatchNormV3g
IdentityIdentityFusedBatchNormV3:y:0*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:0::::0:N J
&
_output_shapes
:0
 
_user_specified_nameinputs: 

_output_shapes
:0
ฤ
u
'__inference_conv2d_7_layer_call_fn_6458

inputs
conv2d_7_kernel
identityขStatefulPartitionedCall์
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_7_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:`*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_7_layer_call_and_return_conditional_losses_30602
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:`2

Identity"
identityIdentity:output:0*)
_input_shapes
:@:22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:@
 
_user_specified_nameinputs
๘
?
L__inference_batch_normalization_layer_call_and_return_conditional_losses_981

inputs+
'readvariableop_batch_normalization_betaC
?fusedbatchnormv3_readvariableop_batch_normalization_moving_meanI
Efusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance.
*fusedbatchnormv3_batch_normalization_const
identityขAssignNewValueขAssignNewValue_1
ReadVariableOpReadVariableOp'readvariableop_batch_normalization_beta*
_output_shapes
: *
dtype02
ReadVariableOpพ
FusedBatchNormV3/ReadVariableOpReadVariableOp?fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpศ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpEfusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputs*fusedbatchnormv3_batch_normalization_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o:*
exponential_avg_factor%
ื#<2
FusedBatchNormV3ญ
AssignNewValueAssignVariableOp?fusedbatchnormv3_readvariableop_batch_normalization_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueร
AssignNewValue_1AssignVariableOpEfusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*X
_classN
LJloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1ฆ
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:+??????????????????????????? :::: 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs: 

_output_shapes
: 
ํ

1__inference_depthwise_conv2d_2_layer_call_fn_1716

inputs'
#depthwise_conv2d_2_depthwise_kernel
identityขStatefulPartitionedCallฅ
StatefulPartitionedCallStatefulPartitionedCallinputs#depthwise_conv2d_2_depthwise_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_depthwise_conv2d_2_layer_call_and_return_conditional_losses_17122
StatefulPartitionedCallจ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+???????????????????????????0:22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
บ

$__inference_dense_layer_call_fn_6863

inputs
dense_kernel

dense_bias
identityขStatefulPartitionedCall์
StatefulPartitionedCallStatefulPartitionedCallinputsdense_kernel
dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_33552
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*%
_input_shapes
:@::22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes

:@
 
_user_specified_nameinputs
โ
ใ
O__inference_batch_normalization_9_layer_call_and_return_conditional_losses_3029

inputs.
*readvariableop_batch_normalization_9_gamma/
+readvariableop_1_batch_normalization_9_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_9_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_9_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_9_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1ภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_9_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ม
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3g
IdentityIdentityFusedBatchNormV3:y:0*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*5
_input_shapes$
":@:::::N J
&
_output_shapes
:@
 
_user_specified_nameinputs
ง

O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_5865

inputs-
)readvariableop_batch_normalization_5_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_5_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance0
,fusedbatchnormv3_batch_normalization_5_const
identityขAssignNewValueขAssignNewValue_1
ReadVariableOpReadVariableOp)readvariableop_batch_normalization_5_beta*
_output_shapes
:0*
dtype02
ReadVariableOpภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_5_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ใ
FusedBatchNormV3FusedBatchNormV3inputs,fusedbatchnormv3_batch_normalization_5_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
ื#<2
FusedBatchNormV3ฑ
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_5_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_5/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueว
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_5/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:0::::02 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:N J
&
_output_shapes
:0
 
_user_specified_nameinputs: 

_output_shapes
:0
	
ญ
L__inference_depthwise_conv2d_3_layer_call_and_return_conditional_losses_2006

inputs@
<depthwise_readvariableop_depthwise_conv2d_3_depthwise_kernel
identityน
depthwise/ReadVariableOpReadVariableOp<depthwise_readvariableop_depthwise_conv2d_3_depthwise_kernel*&
_output_shapes
:`*
dtype02
depthwise/ReadVariableOp{
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      `      2
depthwise/Shape
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
depthwise/dilation_rateอ
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????`*
paddingSAME*
strides
2
	depthwise
IdentityIdentitydepthwise:output:0*
T0*A
_output_shapes/
-:+???????????????????????????`2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+???????????????????????????`::i e
A
_output_shapes/
-:+???????????????????????????`
 
_user_specified_nameinputs
๎	

4__inference_batch_normalization_1_layer_call_fn_5403

inputs
batch_normalization_1_beta%
!batch_normalization_1_moving_mean)
%batch_normalization_1_moving_variance
batch_normalization_1_const
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_1_beta!batch_normalization_1_moving_mean%batch_normalization_1_moving_variancebatch_normalization_1_const*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_10692
StatefulPartitionedCallจ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:+???????????????????????????0::::022
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs: 

_output_shapes
:0
ฏ
]
A__inference_re_lu_5_layer_call_and_return_conditional_losses_6205

inputs
identityP
Relu6Relu6inputs*
T0*&
_output_shapes
:02
Relu6f
IdentityIdentityRelu6:activations:0*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*%
_input_shapes
:0:N J
&
_output_shapes
:0
 
_user_specified_nameinputs
ษ

B__inference_conv2d_2_layer_call_and_return_conditional_losses_5595

inputs)
%conv2d_readvariableop_conv2d_2_kernel
identity
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_2_kernel*&
_output_shapes
:0 *
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingSAME*
strides
2
Conv2Db
IdentityIdentityConv2D:output:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*)
_input_shapes
:0::N J
&
_output_shapes
:0
 
_user_specified_nameinputs

่
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_6492

inputs.
*readvariableop_batch_normalization_10_betaF
Bfusedbatchnormv3_readvariableop_batch_normalization_10_moving_meanL
Hfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance1
-fusedbatchnormv3_batch_normalization_10_const
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_10_beta*
_output_shapes
:`*
dtype02
ReadVariableOpม
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_10_moving_mean*
_output_shapes
:`*
dtype02!
FusedBatchNormV3/ReadVariableOpห
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance*
_output_shapes
:`*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1๑
FusedBatchNormV3FusedBatchNormV3inputs-fusedbatchnormv3_batch_normalization_10_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????`:`:`:`:`:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????`2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:+???????????????????????????`::::`:i e
A
_output_shapes/
-:+???????????????????????????`
 
_user_specified_nameinputs: 

_output_shapes
:`
	

4__inference_batch_normalization_8_layer_call_fn_6262

inputs
batch_normalization_8_beta%
!batch_normalization_8_moving_mean)
%batch_normalization_8_moving_variance
batch_normalization_8_const
identityขStatefulPartitionedCall๐
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_8_beta!batch_normalization_8_moving_mean%batch_normalization_8_moving_variancebatch_normalization_8_const*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_29402
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:0::::022
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:0
 
_user_specified_nameinputs: 

_output_shapes
:0


O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5491

inputs-
)readvariableop_batch_normalization_2_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_2_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance0
,fusedbatchnormv3_batch_normalization_2_const
identityขAssignNewValueขAssignNewValue_1
ReadVariableOpReadVariableOp)readvariableop_batch_normalization_2_beta*
_output_shapes
:0*
dtype02
ReadVariableOpภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputs,fusedbatchnormv3_batch_normalization_2_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
ื#<2
FusedBatchNormV3ฑ
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_2_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_2/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueว
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_2/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1ฆ
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:+???????????????????????????0::::02 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs: 

_output_shapes
:0
๗
ใ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5508

inputs-
)readvariableop_batch_normalization_2_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_2_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance0
,fusedbatchnormv3_batch_normalization_2_const
identity
ReadVariableOpReadVariableOp)readvariableop_batch_normalization_2_beta*
_output_shapes
:0*
dtype02
ReadVariableOpภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1๐
FusedBatchNormV3FusedBatchNormV3inputs,fusedbatchnormv3_batch_normalization_2_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:+???????????????????????????0::::0:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs: 

_output_shapes
:0

่
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_1977

inputs.
*readvariableop_batch_normalization_10_betaF
Bfusedbatchnormv3_readvariableop_batch_normalization_10_moving_meanL
Hfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance1
-fusedbatchnormv3_batch_normalization_10_const
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_10_beta*
_output_shapes
:`*
dtype02
ReadVariableOpม
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_10_moving_mean*
_output_shapes
:`*
dtype02!
FusedBatchNormV3/ReadVariableOpห
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance*
_output_shapes
:`*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1๑
FusedBatchNormV3FusedBatchNormV3inputs-fusedbatchnormv3_batch_normalization_10_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????`:`:`:`:`:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????`2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:+???????????????????????????`::::`:i e
A
_output_shapes/
-:+???????????????????????????`
 
_user_specified_nameinputs: 

_output_shapes
:`
ด

P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_6527

inputs.
*readvariableop_batch_normalization_10_betaF
Bfusedbatchnormv3_readvariableop_batch_normalization_10_moving_meanL
Hfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance1
-fusedbatchnormv3_batch_normalization_10_const
identityขAssignNewValueขAssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_10_beta*
_output_shapes
:`*
dtype02
ReadVariableOpม
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_10_moving_mean*
_output_shapes
:`*
dtype02!
FusedBatchNormV3/ReadVariableOpห
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance*
_output_shapes
:`*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ไ
FusedBatchNormV3FusedBatchNormV3inputs-fusedbatchnormv3_batch_normalization_10_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:`:`:`:`:`:*
epsilon%o:*
exponential_avg_factor%
ื#<2
FusedBatchNormV3ณ
AssignNewValueAssignVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_10_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*U
_classK
IGloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_10/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueษ
AssignNewValue_1AssignVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*[
_classQ
OMloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_10/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*&
_output_shapes
:`2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:`::::`2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:N J
&
_output_shapes
:`
 
_user_specified_nameinputs: 

_output_shapes
:`
ฆ
ญ
?__inference_dense_layer_call_and_return_conditional_losses_3355

inputs&
"matmul_readvariableop_dense_kernel%
!biasadd_readvariableop_dense_bias
identity
MatMul/ReadVariableOpReadVariableOp"matmul_readvariableop_dense_kernel*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOpj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
MatMul
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_dense_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2	
BiasAdd[
IdentityIdentityBiasAdd:output:0*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*%
_input_shapes
:@:::F B

_output_shapes

:@
 
_user_specified_nameinputs

่
P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_2091

inputs.
*readvariableop_batch_normalization_11_betaF
Bfusedbatchnormv3_readvariableop_batch_normalization_11_moving_meanL
Hfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance1
-fusedbatchnormv3_batch_normalization_11_const
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_11_beta*
_output_shapes
:`*
dtype02
ReadVariableOpม
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_11_moving_mean*
_output_shapes
:`*
dtype02!
FusedBatchNormV3/ReadVariableOpห
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance*
_output_shapes
:`*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1๑
FusedBatchNormV3FusedBatchNormV3inputs-fusedbatchnormv3_batch_normalization_11_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????`:`:`:`:`:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????`2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:+???????????????????????????`::::`:i e
A
_output_shapes/
-:+???????????????????????????`
 
_user_specified_nameinputs: 

_output_shapes
:`
?
_
A__inference_dropout_layer_call_and_return_conditional_losses_6836

inputs

identity_1Q
IdentityIdentityinputs*
T0*
_output_shapes

:@2

Identity`

Identity_1IdentityIdentity:output:0*
T0*
_output_shapes

:@2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes

:@:F B

_output_shapes

:@
 
_user_specified_nameinputs
ง

O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5543

inputs-
)readvariableop_batch_normalization_2_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_2_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance0
,fusedbatchnormv3_batch_normalization_2_const
identityขAssignNewValueขAssignNewValue_1
ReadVariableOpReadVariableOp)readvariableop_batch_normalization_2_beta*
_output_shapes
:0*
dtype02
ReadVariableOpภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ใ
FusedBatchNormV3FusedBatchNormV3inputs,fusedbatchnormv3_batch_normalization_2_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
ื#<2
FusedBatchNormV3ฑ
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_2_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_2/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueว
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_2/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:0::::02 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:N J
&
_output_shapes
:0
 
_user_specified_nameinputs: 

_output_shapes
:0
?

O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_5993

inputs.
*readvariableop_batch_normalization_6_gamma/
+readvariableop_1_batch_normalization_6_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_6_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance
identityขAssignNewValueขAssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_6_gamma*
_output_shapes
: *
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_6_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1ภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_6_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ฯ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
exponential_avg_factor%
ื#<2
FusedBatchNormV3ฑ
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_6_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_6/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueว
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_6/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:N J
&
_output_shapes
: 
 
_user_specified_nameinputs
ฤ
u
'__inference_conv2d_8_layer_call_fn_6699

inputs
conv2d_8_kernel
identityขStatefulPartitionedCall์
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_8_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_8_layer_call_and_return_conditional_losses_32252
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*)
_input_shapes
:`:22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:`
 
_user_specified_nameinputs
อ	

4__inference_batch_normalization_3_layer_call_fn_5709

inputs
batch_normalization_3_gamma
batch_normalization_3_beta%
!batch_normalization_3_moving_mean)
%batch_normalization_3_moving_variance
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_3_gammabatch_normalization_3_beta!batch_normalization_3_moving_mean%batch_normalization_3_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_13012
StatefulPartitionedCallจ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
น-
?
+__inference_functional_1_layer_call_fn_5137

inputs
conv2d_kernel
batch_normalization_beta#
batch_normalization_moving_mean'
#batch_normalization_moving_variance
batch_normalization_const
conv2d_1_kernel
batch_normalization_1_beta%
!batch_normalization_1_moving_mean)
%batch_normalization_1_moving_variance
batch_normalization_1_const%
!depthwise_conv2d_depthwise_kernel
batch_normalization_2_beta%
!batch_normalization_2_moving_mean)
%batch_normalization_2_moving_variance
batch_normalization_2_const
conv2d_2_kernel
batch_normalization_3_gamma
batch_normalization_3_beta%
!batch_normalization_3_moving_mean)
%batch_normalization_3_moving_variance
conv2d_3_kernel
batch_normalization_4_beta%
!batch_normalization_4_moving_mean)
%batch_normalization_4_moving_variance
batch_normalization_4_const'
#depthwise_conv2d_1_depthwise_kernel
batch_normalization_5_beta%
!batch_normalization_5_moving_mean)
%batch_normalization_5_moving_variance
batch_normalization_5_const
conv2d_4_kernel
batch_normalization_6_gamma
batch_normalization_6_beta%
!batch_normalization_6_moving_mean)
%batch_normalization_6_moving_variance
conv2d_5_kernel
batch_normalization_7_beta%
!batch_normalization_7_moving_mean)
%batch_normalization_7_moving_variance
batch_normalization_7_const'
#depthwise_conv2d_2_depthwise_kernel
batch_normalization_8_beta%
!batch_normalization_8_moving_mean)
%batch_normalization_8_moving_variance
batch_normalization_8_const
conv2d_6_kernel
batch_normalization_9_gamma
batch_normalization_9_beta%
!batch_normalization_9_moving_mean)
%batch_normalization_9_moving_variance
conv2d_7_kernel
batch_normalization_10_beta&
"batch_normalization_10_moving_mean*
&batch_normalization_10_moving_variance 
batch_normalization_10_const'
#depthwise_conv2d_3_depthwise_kernel
batch_normalization_11_beta&
"batch_normalization_11_moving_mean*
&batch_normalization_11_moving_variance 
batch_normalization_11_const
conv2d_8_kernel 
batch_normalization_12_gamma
batch_normalization_12_beta&
"batch_normalization_12_moving_mean*
&batch_normalization_12_moving_variance
dense_kernel

dense_bias
identityขStatefulPartitionedCallฆ
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_kernelbatch_normalization_betabatch_normalization_moving_mean#batch_normalization_moving_variancebatch_normalization_constconv2d_1_kernelbatch_normalization_1_beta!batch_normalization_1_moving_mean%batch_normalization_1_moving_variancebatch_normalization_1_const!depthwise_conv2d_depthwise_kernelbatch_normalization_2_beta!batch_normalization_2_moving_mean%batch_normalization_2_moving_variancebatch_normalization_2_constconv2d_2_kernelbatch_normalization_3_gammabatch_normalization_3_beta!batch_normalization_3_moving_mean%batch_normalization_3_moving_varianceconv2d_3_kernelbatch_normalization_4_beta!batch_normalization_4_moving_mean%batch_normalization_4_moving_variancebatch_normalization_4_const#depthwise_conv2d_1_depthwise_kernelbatch_normalization_5_beta!batch_normalization_5_moving_mean%batch_normalization_5_moving_variancebatch_normalization_5_constconv2d_4_kernelbatch_normalization_6_gammabatch_normalization_6_beta!batch_normalization_6_moving_mean%batch_normalization_6_moving_varianceconv2d_5_kernelbatch_normalization_7_beta!batch_normalization_7_moving_mean%batch_normalization_7_moving_variancebatch_normalization_7_const#depthwise_conv2d_2_depthwise_kernelbatch_normalization_8_beta!batch_normalization_8_moving_mean%batch_normalization_8_moving_variancebatch_normalization_8_constconv2d_6_kernelbatch_normalization_9_gammabatch_normalization_9_beta!batch_normalization_9_moving_mean%batch_normalization_9_moving_varianceconv2d_7_kernelbatch_normalization_10_beta"batch_normalization_10_moving_mean&batch_normalization_10_moving_variancebatch_normalization_10_const#depthwise_conv2d_3_depthwise_kernelbatch_normalization_11_beta"batch_normalization_11_moving_mean&batch_normalization_11_moving_variancebatch_normalization_11_constconv2d_8_kernelbatch_normalization_12_gammabatch_normalization_12_beta"batch_normalization_12_moving_mean&batch_normalization_12_moving_variancedense_kernel
dense_bias*O
TinH
F2D*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*B
_read_only_resource_inputs$
"  !$%)*./03489=>?BC*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_functional_1_layer_call_and_return_conditional_losses_35952
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*ส
_input_shapesธ
ต:?????????1(::::: :::::0:::::0::::::::::0:::::0::::::::::0:::::0::::::::::`:::::`:::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????1(
 
_user_specified_nameinputs: 

_output_shapes
: : 


_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0: (

_output_shapes
:0: -

_output_shapes
:0: 7

_output_shapes
:`: <

_output_shapes
:`
ห	

4__inference_batch_normalization_6_layer_call_fn_6074

inputs
batch_normalization_6_gamma
batch_normalization_6_beta%
!batch_normalization_6_moving_mean)
%batch_normalization_6_moving_variance
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_6_gammabatch_normalization_6_beta!batch_normalization_6_moving_mean%batch_normalization_6_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_15682
StatefulPartitionedCallจ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?

O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2770

inputs.
*readvariableop_batch_normalization_6_gamma/
+readvariableop_1_batch_normalization_6_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_6_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance
identityขAssignNewValueขAssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_6_gamma*
_output_shapes
: *
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_6_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1ภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_6_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ฯ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
exponential_avg_factor%
ื#<2
FusedBatchNormV3ฑ
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_6_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_6/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueว
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_6/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:N J
&
_output_shapes
: 
 
_user_specified_nameinputs
ฯ
ใ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1595

inputs.
*readvariableop_batch_normalization_6_gamma/
+readvariableop_1_batch_normalization_6_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_6_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_6_gamma*
_output_shapes
: *
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_6_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1ภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_6_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? :::::i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs


O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1771

inputs-
)readvariableop_batch_normalization_8_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_8_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance0
,fusedbatchnormv3_batch_normalization_8_const
identityขAssignNewValueขAssignNewValue_1
ReadVariableOpReadVariableOp)readvariableop_batch_normalization_8_beta*
_output_shapes
:0*
dtype02
ReadVariableOpภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputs,fusedbatchnormv3_batch_normalization_8_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
ื#<2
FusedBatchNormV3ฑ
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_8_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_8/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueว
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_8/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1ฆ
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:+???????????????????????????0::::02 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs: 

_output_shapes
:0
-
ึ
"__inference_signature_wrapper_3923
input_1
conv2d_kernel
batch_normalization_beta#
batch_normalization_moving_mean'
#batch_normalization_moving_variance
batch_normalization_const
conv2d_1_kernel
batch_normalization_1_beta%
!batch_normalization_1_moving_mean)
%batch_normalization_1_moving_variance
batch_normalization_1_const%
!depthwise_conv2d_depthwise_kernel
batch_normalization_2_beta%
!batch_normalization_2_moving_mean)
%batch_normalization_2_moving_variance
batch_normalization_2_const
conv2d_2_kernel
batch_normalization_3_gamma
batch_normalization_3_beta%
!batch_normalization_3_moving_mean)
%batch_normalization_3_moving_variance
conv2d_3_kernel
batch_normalization_4_beta%
!batch_normalization_4_moving_mean)
%batch_normalization_4_moving_variance
batch_normalization_4_const'
#depthwise_conv2d_1_depthwise_kernel
batch_normalization_5_beta%
!batch_normalization_5_moving_mean)
%batch_normalization_5_moving_variance
batch_normalization_5_const
conv2d_4_kernel
batch_normalization_6_gamma
batch_normalization_6_beta%
!batch_normalization_6_moving_mean)
%batch_normalization_6_moving_variance
conv2d_5_kernel
batch_normalization_7_beta%
!batch_normalization_7_moving_mean)
%batch_normalization_7_moving_variance
batch_normalization_7_const'
#depthwise_conv2d_2_depthwise_kernel
batch_normalization_8_beta%
!batch_normalization_8_moving_mean)
%batch_normalization_8_moving_variance
batch_normalization_8_const
conv2d_6_kernel
batch_normalization_9_gamma
batch_normalization_9_beta%
!batch_normalization_9_moving_mean)
%batch_normalization_9_moving_variance
conv2d_7_kernel
batch_normalization_10_beta&
"batch_normalization_10_moving_mean*
&batch_normalization_10_moving_variance 
batch_normalization_10_const'
#depthwise_conv2d_3_depthwise_kernel
batch_normalization_11_beta&
"batch_normalization_11_moving_mean*
&batch_normalization_11_moving_variance 
batch_normalization_11_const
conv2d_8_kernel 
batch_normalization_12_gamma
batch_normalization_12_beta&
"batch_normalization_12_moving_mean*
&batch_normalization_12_moving_variance
dense_kernel

dense_bias
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_kernelbatch_normalization_betabatch_normalization_moving_mean#batch_normalization_moving_variancebatch_normalization_constconv2d_1_kernelbatch_normalization_1_beta!batch_normalization_1_moving_mean%batch_normalization_1_moving_variancebatch_normalization_1_const!depthwise_conv2d_depthwise_kernelbatch_normalization_2_beta!batch_normalization_2_moving_mean%batch_normalization_2_moving_variancebatch_normalization_2_constconv2d_2_kernelbatch_normalization_3_gammabatch_normalization_3_beta!batch_normalization_3_moving_mean%batch_normalization_3_moving_varianceconv2d_3_kernelbatch_normalization_4_beta!batch_normalization_4_moving_mean%batch_normalization_4_moving_variancebatch_normalization_4_const#depthwise_conv2d_1_depthwise_kernelbatch_normalization_5_beta!batch_normalization_5_moving_mean%batch_normalization_5_moving_variancebatch_normalization_5_constconv2d_4_kernelbatch_normalization_6_gammabatch_normalization_6_beta!batch_normalization_6_moving_mean%batch_normalization_6_moving_varianceconv2d_5_kernelbatch_normalization_7_beta!batch_normalization_7_moving_mean%batch_normalization_7_moving_variancebatch_normalization_7_const#depthwise_conv2d_2_depthwise_kernelbatch_normalization_8_beta!batch_normalization_8_moving_mean%batch_normalization_8_moving_variancebatch_normalization_8_constconv2d_6_kernelbatch_normalization_9_gammabatch_normalization_9_beta!batch_normalization_9_moving_mean%batch_normalization_9_moving_varianceconv2d_7_kernelbatch_normalization_10_beta"batch_normalization_10_moving_mean&batch_normalization_10_moving_variancebatch_normalization_10_const#depthwise_conv2d_3_depthwise_kernelbatch_normalization_11_beta"batch_normalization_11_moving_mean&batch_normalization_11_moving_variancebatch_normalization_11_constconv2d_8_kernelbatch_normalization_12_gammabatch_normalization_12_beta"batch_normalization_12_moving_mean&batch_normalization_12_moving_variancedense_kernel
dense_bias*O
TinH
F2D*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*\
_read_only_resource_inputs>
<:	 !"#$%&')*+,./012345689:;=>?@ABC*-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__wrapped_model_9262
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*ม
_input_shapesฏ
ฌ:1(::::: :::::0:::::0::::::::::0:::::0::::::::::0:::::0::::::::::`:::::`:::::::22
StatefulPartitionedCallStatefulPartitionedCall:K G
"
_output_shapes
:1(
!
_user_specified_name	input_1: 

_output_shapes
: : 


_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0: (

_output_shapes
:0: -

_output_shapes
:0: 7

_output_shapes
:`: <

_output_shapes
:`
	
ญ
L__inference_depthwise_conv2d_2_layer_call_and_return_conditional_losses_1700

inputs@
<depthwise_readvariableop_depthwise_conv2d_2_depthwise_kernel
identityน
depthwise/ReadVariableOpReadVariableOp<depthwise_readvariableop_depthwise_conv2d_2_depthwise_kernel*&
_output_shapes
:0*
dtype02
depthwise/ReadVariableOp{
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      0      2
depthwise/Shape
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
depthwise/dilation_rateอ
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????0*
paddingSAME*
strides
2
	depthwise
IdentityIdentitydepthwise:output:0*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+???????????????????????????0::i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
๘	

5__inference_batch_normalization_11_layer_call_fn_6667

inputs
batch_normalization_11_beta&
"batch_normalization_11_moving_mean*
&batch_normalization_11_moving_variance 
batch_normalization_11_const
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_11_beta"batch_normalization_11_moving_mean&batch_normalization_11_moving_variancebatch_normalization_11_const*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????`*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_20652
StatefulPartitionedCallจ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????`2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:+???????????????????????????`::::`22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????`
 
_user_specified_nameinputs: 

_output_shapes
:`
?

O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_5619

inputs.
*readvariableop_batch_normalization_3_gamma/
+readvariableop_1_batch_normalization_3_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_3_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance
identityขAssignNewValueขAssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_3_gamma*
_output_shapes
: *
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_3_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1ภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ฯ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
exponential_avg_factor%
ื#<2
FusedBatchNormV3ฑ
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_3_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_3/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueว
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_3/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:N J
&
_output_shapes
: 
 
_user_specified_nameinputs
	
ญ
L__inference_depthwise_conv2d_3_layer_call_and_return_conditional_losses_1994

inputs@
<depthwise_readvariableop_depthwise_conv2d_3_depthwise_kernel
identityน
depthwise/ReadVariableOpReadVariableOp<depthwise_readvariableop_depthwise_conv2d_3_depthwise_kernel*&
_output_shapes
:`*
dtype02
depthwise/ReadVariableOp{
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      `      2
depthwise/Shape
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
depthwise/dilation_rateอ
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????`*
paddingSAME*
strides
2
	depthwise
IdentityIdentitydepthwise:output:0*
T0*A
_output_shapes/
-:+???????????????????????????`2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+???????????????????????????`::i e
A
_output_shapes/
-:+???????????????????????????`
 
_user_specified_nameinputs
๎	

4__inference_batch_normalization_5_layer_call_fn_5943

inputs
batch_normalization_5_beta%
!batch_normalization_5_moving_mean)
%batch_normalization_5_moving_variance
batch_normalization_5_const
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_5_beta!batch_normalization_5_moving_mean%batch_normalization_5_moving_variancebatch_normalization_5_const*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_14772
StatefulPartitionedCallจ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:+???????????????????????????0::::022
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs: 

_output_shapes
:0
ษ

B__inference_conv2d_1_layer_call_and_return_conditional_losses_2322

inputs)
%conv2d_readvariableop_conv2d_1_kernel
identity
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_1_kernel*&
_output_shapes
: 0*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:0*
paddingSAME*
strides
2
Conv2Db
IdentityIdentityConv2D:output:0*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*)
_input_shapes
: ::N J
&
_output_shapes
: 
 
_user_specified_nameinputs
ๅ
ู
M__inference_batch_normalization_layer_call_and_return_conditional_losses_5267

inputs+
'readvariableop_batch_normalization_betaC
?fusedbatchnormv3_readvariableop_batch_normalization_moving_meanI
Efusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance.
*fusedbatchnormv3_batch_normalization_const
identity
ReadVariableOpReadVariableOp'readvariableop_batch_normalization_beta*
_output_shapes
: *
dtype02
ReadVariableOpพ
FusedBatchNormV3/ReadVariableOpReadVariableOp?fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpศ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpEfusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1๎
FusedBatchNormV3FusedBatchNormV3inputs*fusedbatchnormv3_batch_normalization_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:+??????????????????????????? :::: :i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs: 

_output_shapes
: 
	
ญ
L__inference_depthwise_conv2d_1_layer_call_and_return_conditional_losses_1406

inputs@
<depthwise_readvariableop_depthwise_conv2d_1_depthwise_kernel
identityน
depthwise/ReadVariableOpReadVariableOp<depthwise_readvariableop_depthwise_conv2d_1_depthwise_kernel*&
_output_shapes
:0*
dtype02
depthwise/ReadVariableOp{
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      0      2
depthwise/Shape
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
depthwise/dilation_rateอ
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????0*
paddingSAME*
strides
2
	depthwise
IdentityIdentitydepthwise:output:0*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+???????????????????????????0::i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
ฏ
]
A__inference_re_lu_3_layer_call_and_return_conditional_losses_2650

inputs
identityP
Relu6Relu6inputs*
T0*&
_output_shapes
:02
Relu6f
IdentityIdentityRelu6:activations:0*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*%
_input_shapes
:0:N J
&
_output_shapes
:0
 
_user_specified_nameinputs


O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5803

inputs-
)readvariableop_batch_normalization_4_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_4_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance0
,fusedbatchnormv3_batch_normalization_4_const
identityขAssignNewValueขAssignNewValue_1
ReadVariableOpReadVariableOp)readvariableop_batch_normalization_4_beta*
_output_shapes
:0*
dtype02
ReadVariableOpภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputs,fusedbatchnormv3_batch_normalization_4_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
ื#<2
FusedBatchNormV3ฑ
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_4_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_4/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueว
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_4/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1ฆ
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:+???????????????????????????0::::02 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs: 

_output_shapes
:0
๗
ใ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1503

inputs-
)readvariableop_batch_normalization_5_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_5_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance0
,fusedbatchnormv3_batch_normalization_5_const
identity
ReadVariableOpReadVariableOp)readvariableop_batch_normalization_5_beta*
_output_shapes
:0*
dtype02
ReadVariableOpภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_5_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1๐
FusedBatchNormV3FusedBatchNormV3inputs,fusedbatchnormv3_batch_normalization_5_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:+???????????????????????????0::::0:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs: 

_output_shapes
:0

P
$__inference_add_1_layer_call_fn_6819
inputs_0
inputs_1
identityษ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_add_1_layer_call_and_return_conditional_losses_33002
PartitionedCallk
IdentityIdentityPartitionedCall:output:0*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:@:@:P L
&
_output_shapes
:@
"
_user_specified_name
inputs/0:PL
&
_output_shapes
:@
"
_user_specified_name
inputs/1

ใ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2443

inputs-
)readvariableop_batch_normalization_2_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_2_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance0
,fusedbatchnormv3_batch_normalization_2_const
identity
ReadVariableOpReadVariableOp)readvariableop_batch_normalization_2_beta*
_output_shapes
:0*
dtype02
ReadVariableOpภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ี
FusedBatchNormV3FusedBatchNormV3inputs,fusedbatchnormv3_batch_normalization_2_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:0:0:0:0:0:*
epsilon%o:*
is_training( 2
FusedBatchNormV3g
IdentityIdentityFusedBatchNormV3:y:0*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:0::::0:N J
&
_output_shapes
:0
 
_user_specified_nameinputs: 

_output_shapes
:0
่
B
&__inference_dropout_layer_call_fn_6846

inputs
identityถ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_33322
PartitionedCallc
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes

:@2

Identity"
identityIdentity:output:0*
_input_shapes

:@:F B

_output_shapes

:@
 
_user_specified_nameinputs
เ
ั)
F__inference_functional_1_layer_call_and_return_conditional_losses_5065

inputs.
*conv2d_conv2d_readvariableop_conv2d_kernel?
;batch_normalization_readvariableop_batch_normalization_betaW
Sbatch_normalization_fusedbatchnormv3_readvariableop_batch_normalization_moving_mean]
Ybatch_normalization_fusedbatchnormv3_readvariableop_1_batch_normalization_moving_varianceB
>batch_normalization_fusedbatchnormv3_batch_normalization_const2
.conv2d_1_conv2d_readvariableop_conv2d_1_kernelC
?batch_normalization_1_readvariableop_batch_normalization_1_beta[
Wbatch_normalization_1_fusedbatchnormv3_readvariableop_batch_normalization_1_moving_meana
]batch_normalization_1_fusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_varianceF
Bbatch_normalization_1_fusedbatchnormv3_batch_normalization_1_constO
Kdepthwise_conv2d_depthwise_readvariableop_depthwise_conv2d_depthwise_kernelC
?batch_normalization_2_readvariableop_batch_normalization_2_beta[
Wbatch_normalization_2_fusedbatchnormv3_readvariableop_batch_normalization_2_moving_meana
]batch_normalization_2_fusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_varianceF
Bbatch_normalization_2_fusedbatchnormv3_batch_normalization_2_const2
.conv2d_2_conv2d_readvariableop_conv2d_2_kernelD
@batch_normalization_3_readvariableop_batch_normalization_3_gammaE
Abatch_normalization_3_readvariableop_1_batch_normalization_3_beta[
Wbatch_normalization_3_fusedbatchnormv3_readvariableop_batch_normalization_3_moving_meana
]batch_normalization_3_fusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance2
.conv2d_3_conv2d_readvariableop_conv2d_3_kernelC
?batch_normalization_4_readvariableop_batch_normalization_4_beta[
Wbatch_normalization_4_fusedbatchnormv3_readvariableop_batch_normalization_4_moving_meana
]batch_normalization_4_fusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_varianceF
Bbatch_normalization_4_fusedbatchnormv3_batch_normalization_4_constS
Odepthwise_conv2d_1_depthwise_readvariableop_depthwise_conv2d_1_depthwise_kernelC
?batch_normalization_5_readvariableop_batch_normalization_5_beta[
Wbatch_normalization_5_fusedbatchnormv3_readvariableop_batch_normalization_5_moving_meana
]batch_normalization_5_fusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_varianceF
Bbatch_normalization_5_fusedbatchnormv3_batch_normalization_5_const2
.conv2d_4_conv2d_readvariableop_conv2d_4_kernelD
@batch_normalization_6_readvariableop_batch_normalization_6_gammaE
Abatch_normalization_6_readvariableop_1_batch_normalization_6_beta[
Wbatch_normalization_6_fusedbatchnormv3_readvariableop_batch_normalization_6_moving_meana
]batch_normalization_6_fusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance2
.conv2d_5_conv2d_readvariableop_conv2d_5_kernelC
?batch_normalization_7_readvariableop_batch_normalization_7_beta[
Wbatch_normalization_7_fusedbatchnormv3_readvariableop_batch_normalization_7_moving_meana
]batch_normalization_7_fusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_varianceF
Bbatch_normalization_7_fusedbatchnormv3_batch_normalization_7_constS
Odepthwise_conv2d_2_depthwise_readvariableop_depthwise_conv2d_2_depthwise_kernelC
?batch_normalization_8_readvariableop_batch_normalization_8_beta[
Wbatch_normalization_8_fusedbatchnormv3_readvariableop_batch_normalization_8_moving_meana
]batch_normalization_8_fusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_varianceF
Bbatch_normalization_8_fusedbatchnormv3_batch_normalization_8_const2
.conv2d_6_conv2d_readvariableop_conv2d_6_kernelD
@batch_normalization_9_readvariableop_batch_normalization_9_gammaE
Abatch_normalization_9_readvariableop_1_batch_normalization_9_beta[
Wbatch_normalization_9_fusedbatchnormv3_readvariableop_batch_normalization_9_moving_meana
]batch_normalization_9_fusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance2
.conv2d_7_conv2d_readvariableop_conv2d_7_kernelE
Abatch_normalization_10_readvariableop_batch_normalization_10_beta]
Ybatch_normalization_10_fusedbatchnormv3_readvariableop_batch_normalization_10_moving_meanc
_batch_normalization_10_fusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_varianceH
Dbatch_normalization_10_fusedbatchnormv3_batch_normalization_10_constS
Odepthwise_conv2d_3_depthwise_readvariableop_depthwise_conv2d_3_depthwise_kernelE
Abatch_normalization_11_readvariableop_batch_normalization_11_beta]
Ybatch_normalization_11_fusedbatchnormv3_readvariableop_batch_normalization_11_moving_meanc
_batch_normalization_11_fusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_varianceH
Dbatch_normalization_11_fusedbatchnormv3_batch_normalization_11_const2
.conv2d_8_conv2d_readvariableop_conv2d_8_kernelF
Bbatch_normalization_12_readvariableop_batch_normalization_12_gammaG
Cbatch_normalization_12_readvariableop_1_batch_normalization_12_beta]
Ybatch_normalization_12_fusedbatchnormv3_readvariableop_batch_normalization_12_moving_meanc
_batch_normalization_12_fusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance,
(dense_matmul_readvariableop_dense_kernel+
'dense_biasadd_readvariableop_dense_bias
identity
%tf_op_layer_ExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%tf_op_layer_ExpandDims/ExpandDims/dimฬ
!tf_op_layer_ExpandDims/ExpandDims
ExpandDimsinputs.tf_op_layer_ExpandDims/ExpandDims/dim:output:0*
T0*
_cloned(*&
_output_shapes
:1(2#
!tf_op_layer_ExpandDims/ExpandDimsฏ
conv2d/Conv2D/ReadVariableOpReadVariableOp*conv2d_conv2d_readvariableop_conv2d_kernel*&
_output_shapes
:( *
dtype02
conv2d/Conv2D/ReadVariableOpิ
conv2d/Conv2DConv2D*tf_op_layer_ExpandDims/ExpandDims:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
2
conv2d/Conv2Dภ
"batch_normalization/ReadVariableOpReadVariableOp;batch_normalization_readvariableop_batch_normalization_beta*
_output_shapes
: *
dtype02$
"batch_normalization/ReadVariableOp๚
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpSbatch_normalization_fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
_output_shapes
: *
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpYbatch_normalization_fusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
_output_shapes
: *
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/Conv2D:output:0>batch_normalization_fusedbatchnormv3_batch_normalization_const*batch_normalization/ReadVariableOp:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
is_training( 2&
$batch_normalization/FusedBatchNormV3~
re_lu/Relu6Relu6(batch_normalization/FusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2
re_lu/Relu6ท
conv2d_1/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_conv2d_readvariableop_conv2d_1_kernel*&
_output_shapes
: 0*
dtype02 
conv2d_1/Conv2D/ReadVariableOpศ
conv2d_1/Conv2DConv2Dre_lu/Relu6:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:0*
paddingSAME*
strides
2
conv2d_1/Conv2Dศ
$batch_normalization_1/ReadVariableOpReadVariableOp?batch_normalization_1_readvariableop_batch_normalization_1_beta*
_output_shapes
:0*
dtype02&
$batch_normalization_1/ReadVariableOp
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_1_fusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
:0*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_1_fusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
_output_shapes
:0*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1๋
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/Conv2D:output:0Bbatch_normalization_1_fusedbatchnormv3_batch_normalization_1_const,batch_normalization_1/ReadVariableOp:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:0:0:0:0:0:*
epsilon%o:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3
re_lu_1/Relu6Relu6*batch_normalization_1/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:02
re_lu_1/Relu6๊
)depthwise_conv2d/depthwise/ReadVariableOpReadVariableOpKdepthwise_conv2d_depthwise_readvariableop_depthwise_conv2d_depthwise_kernel*&
_output_shapes
:0*
dtype02+
)depthwise_conv2d/depthwise/ReadVariableOp
 depthwise_conv2d/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      0      2"
 depthwise_conv2d/depthwise/Shapeฅ
(depthwise_conv2d/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2*
(depthwise_conv2d/depthwise/dilation_rate๚
depthwise_conv2d/depthwiseDepthwiseConv2dNativere_lu_1/Relu6:activations:01depthwise_conv2d/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:0*
paddingSAME*
strides
2
depthwise_conv2d/depthwiseศ
$batch_normalization_2/ReadVariableOpReadVariableOp?batch_normalization_2_readvariableop_batch_normalization_2_beta*
_output_shapes
:0*
dtype02&
$batch_normalization_2/ReadVariableOp
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_2_fusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
:0*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_2_fusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
_output_shapes
:0*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1๖
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3#depthwise_conv2d/depthwise:output:0Bbatch_normalization_2_fusedbatchnormv3_batch_normalization_2_const,batch_normalization_2/ReadVariableOp:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:0:0:0:0:0:*
epsilon%o:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3
re_lu_2/Relu6Relu6*batch_normalization_2/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:02
re_lu_2/Relu6ท
conv2d_2/Conv2D/ReadVariableOpReadVariableOp.conv2d_2_conv2d_readvariableop_conv2d_2_kernel*&
_output_shapes
:0 *
dtype02 
conv2d_2/Conv2D/ReadVariableOpส
conv2d_2/Conv2DConv2Dre_lu_2/Relu6:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingSAME*
strides
2
conv2d_2/Conv2Dษ
$batch_normalization_3/ReadVariableOpReadVariableOp@batch_normalization_3_readvariableop_batch_normalization_3_gamma*
_output_shapes
: *
dtype02&
$batch_normalization_3/ReadVariableOpฮ
&batch_normalization_3/ReadVariableOp_1ReadVariableOpAbatch_normalization_3_readvariableop_1_batch_normalization_3_beta*
_output_shapes
: *
dtype02(
&batch_normalization_3/ReadVariableOp_1
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_3_fusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
_output_shapes
: *
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_3_fusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
_output_shapes
: *
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ื
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_2/Conv2D:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3
add/addAddV2re_lu/Relu6:activations:0*batch_normalization_3/FusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2	
add/addท
conv2d_3/Conv2D/ReadVariableOpReadVariableOp.conv2d_3_conv2d_readvariableop_conv2d_3_kernel*&
_output_shapes
: 0*
dtype02 
conv2d_3/Conv2D/ReadVariableOpบ
conv2d_3/Conv2DConv2Dadd/add:z:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:0*
paddingSAME*
strides
2
conv2d_3/Conv2Dศ
$batch_normalization_4/ReadVariableOpReadVariableOp?batch_normalization_4_readvariableop_batch_normalization_4_beta*
_output_shapes
:0*
dtype02&
$batch_normalization_4/ReadVariableOp
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_4_fusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
_output_shapes
:0*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_4_fusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
_output_shapes
:0*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1๋
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_3/Conv2D:output:0Bbatch_normalization_4_fusedbatchnormv3_batch_normalization_4_const,batch_normalization_4/ReadVariableOp:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:0:0:0:0:0:*
epsilon%o:*
is_training( 2(
&batch_normalization_4/FusedBatchNormV3
re_lu_3/Relu6Relu6*batch_normalization_4/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:02
re_lu_3/Relu6๒
+depthwise_conv2d_1/depthwise/ReadVariableOpReadVariableOpOdepthwise_conv2d_1_depthwise_readvariableop_depthwise_conv2d_1_depthwise_kernel*&
_output_shapes
:0*
dtype02-
+depthwise_conv2d_1/depthwise/ReadVariableOpก
"depthwise_conv2d_1/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      0      2$
"depthwise_conv2d_1/depthwise/Shapeฉ
*depthwise_conv2d_1/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2,
*depthwise_conv2d_1/depthwise/dilation_rate
depthwise_conv2d_1/depthwiseDepthwiseConv2dNativere_lu_3/Relu6:activations:03depthwise_conv2d_1/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:0*
paddingSAME*
strides
2
depthwise_conv2d_1/depthwiseศ
$batch_normalization_5/ReadVariableOpReadVariableOp?batch_normalization_5_readvariableop_batch_normalization_5_beta*
_output_shapes
:0*
dtype02&
$batch_normalization_5/ReadVariableOp
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_5_fusedbatchnormv3_readvariableop_batch_normalization_5_moving_mean*
_output_shapes
:0*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_5_fusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance*
_output_shapes
:0*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1๘
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3%depthwise_conv2d_1/depthwise:output:0Bbatch_normalization_5_fusedbatchnormv3_batch_normalization_5_const,batch_normalization_5/ReadVariableOp:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:0:0:0:0:0:*
epsilon%o:*
is_training( 2(
&batch_normalization_5/FusedBatchNormV3
re_lu_4/Relu6Relu6*batch_normalization_5/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:02
re_lu_4/Relu6ท
conv2d_4/Conv2D/ReadVariableOpReadVariableOp.conv2d_4_conv2d_readvariableop_conv2d_4_kernel*&
_output_shapes
:0 *
dtype02 
conv2d_4/Conv2D/ReadVariableOpส
conv2d_4/Conv2DConv2Dre_lu_4/Relu6:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingSAME*
strides
2
conv2d_4/Conv2Dษ
$batch_normalization_6/ReadVariableOpReadVariableOp@batch_normalization_6_readvariableop_batch_normalization_6_gamma*
_output_shapes
: *
dtype02&
$batch_normalization_6/ReadVariableOpฮ
&batch_normalization_6/ReadVariableOp_1ReadVariableOpAbatch_normalization_6_readvariableop_1_batch_normalization_6_beta*
_output_shapes
: *
dtype02(
&batch_normalization_6/ReadVariableOp_1
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_6_fusedbatchnormv3_readvariableop_batch_normalization_6_moving_mean*
_output_shapes
: *
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_6_fusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance*
_output_shapes
: *
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ื
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv2d_4/Conv2D:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
is_training( 2(
&batch_normalization_6/FusedBatchNormV3ท
conv2d_5/Conv2D/ReadVariableOpReadVariableOp.conv2d_5_conv2d_readvariableop_conv2d_5_kernel*&
_output_shapes
: 0*
dtype02 
conv2d_5/Conv2D/ReadVariableOpู
conv2d_5/Conv2DConv2D*batch_normalization_6/FusedBatchNormV3:y:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:0*
paddingSAME*
strides
2
conv2d_5/Conv2Dศ
$batch_normalization_7/ReadVariableOpReadVariableOp?batch_normalization_7_readvariableop_batch_normalization_7_beta*
_output_shapes
:0*
dtype02&
$batch_normalization_7/ReadVariableOp
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_7_fusedbatchnormv3_readvariableop_batch_normalization_7_moving_mean*
_output_shapes
:0*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_7_fusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance*
_output_shapes
:0*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1๋
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3conv2d_5/Conv2D:output:0Bbatch_normalization_7_fusedbatchnormv3_batch_normalization_7_const,batch_normalization_7/ReadVariableOp:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:0:0:0:0:0:*
epsilon%o:*
is_training( 2(
&batch_normalization_7/FusedBatchNormV3
re_lu_5/Relu6Relu6*batch_normalization_7/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:02
re_lu_5/Relu6๒
+depthwise_conv2d_2/depthwise/ReadVariableOpReadVariableOpOdepthwise_conv2d_2_depthwise_readvariableop_depthwise_conv2d_2_depthwise_kernel*&
_output_shapes
:0*
dtype02-
+depthwise_conv2d_2/depthwise/ReadVariableOpก
"depthwise_conv2d_2/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      0      2$
"depthwise_conv2d_2/depthwise/Shapeฉ
*depthwise_conv2d_2/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2,
*depthwise_conv2d_2/depthwise/dilation_rate
depthwise_conv2d_2/depthwiseDepthwiseConv2dNativere_lu_5/Relu6:activations:03depthwise_conv2d_2/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:0*
paddingSAME*
strides
2
depthwise_conv2d_2/depthwiseศ
$batch_normalization_8/ReadVariableOpReadVariableOp?batch_normalization_8_readvariableop_batch_normalization_8_beta*
_output_shapes
:0*
dtype02&
$batch_normalization_8/ReadVariableOp
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_8_fusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean*
_output_shapes
:0*
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_8_fusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance*
_output_shapes
:0*
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1๘
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3%depthwise_conv2d_2/depthwise:output:0Bbatch_normalization_8_fusedbatchnormv3_batch_normalization_8_const,batch_normalization_8/ReadVariableOp:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:0:0:0:0:0:*
epsilon%o:*
is_training( 2(
&batch_normalization_8/FusedBatchNormV3
re_lu_6/Relu6Relu6*batch_normalization_8/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:02
re_lu_6/Relu6ท
conv2d_6/Conv2D/ReadVariableOpReadVariableOp.conv2d_6_conv2d_readvariableop_conv2d_6_kernel*&
_output_shapes
:0@*
dtype02 
conv2d_6/Conv2D/ReadVariableOpส
conv2d_6/Conv2DConv2Dre_lu_6/Relu6:activations:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingSAME*
strides
2
conv2d_6/Conv2Dษ
$batch_normalization_9/ReadVariableOpReadVariableOp@batch_normalization_9_readvariableop_batch_normalization_9_gamma*
_output_shapes
:@*
dtype02&
$batch_normalization_9/ReadVariableOpฮ
&batch_normalization_9/ReadVariableOp_1ReadVariableOpAbatch_normalization_9_readvariableop_1_batch_normalization_9_beta*
_output_shapes
:@*
dtype02(
&batch_normalization_9/ReadVariableOp_1
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_9_fusedbatchnormv3_readvariableop_batch_normalization_9_moving_mean*
_output_shapes
:@*
dtype027
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_9_fusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance*
_output_shapes
:@*
dtype029
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ื
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3conv2d_6/Conv2D:output:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:@:@:@:@:@:*
epsilon%o:*
is_training( 2(
&batch_normalization_9/FusedBatchNormV3ท
conv2d_7/Conv2D/ReadVariableOpReadVariableOp.conv2d_7_conv2d_readvariableop_conv2d_7_kernel*&
_output_shapes
:@`*
dtype02 
conv2d_7/Conv2D/ReadVariableOpู
conv2d_7/Conv2DConv2D*batch_normalization_9/FusedBatchNormV3:y:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:`*
paddingSAME*
strides
2
conv2d_7/Conv2Dฬ
%batch_normalization_10/ReadVariableOpReadVariableOpAbatch_normalization_10_readvariableop_batch_normalization_10_beta*
_output_shapes
:`*
dtype02'
%batch_normalization_10/ReadVariableOp
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOpYbatch_normalization_10_fusedbatchnormv3_readvariableop_batch_normalization_10_moving_mean*
_output_shapes
:`*
dtype028
6batch_normalization_10/FusedBatchNormV3/ReadVariableOp
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp_batch_normalization_10_fusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance*
_output_shapes
:`*
dtype02:
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1๒
'batch_normalization_10/FusedBatchNormV3FusedBatchNormV3conv2d_7/Conv2D:output:0Dbatch_normalization_10_fusedbatchnormv3_batch_normalization_10_const-batch_normalization_10/ReadVariableOp:value:0>batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:`:`:`:`:`:*
epsilon%o:*
is_training( 2)
'batch_normalization_10/FusedBatchNormV3
re_lu_7/Relu6Relu6+batch_normalization_10/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:`2
re_lu_7/Relu6๒
+depthwise_conv2d_3/depthwise/ReadVariableOpReadVariableOpOdepthwise_conv2d_3_depthwise_readvariableop_depthwise_conv2d_3_depthwise_kernel*&
_output_shapes
:`*
dtype02-
+depthwise_conv2d_3/depthwise/ReadVariableOpก
"depthwise_conv2d_3/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      `      2$
"depthwise_conv2d_3/depthwise/Shapeฉ
*depthwise_conv2d_3/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2,
*depthwise_conv2d_3/depthwise/dilation_rate
depthwise_conv2d_3/depthwiseDepthwiseConv2dNativere_lu_7/Relu6:activations:03depthwise_conv2d_3/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:`*
paddingSAME*
strides
2
depthwise_conv2d_3/depthwiseฬ
%batch_normalization_11/ReadVariableOpReadVariableOpAbatch_normalization_11_readvariableop_batch_normalization_11_beta*
_output_shapes
:`*
dtype02'
%batch_normalization_11/ReadVariableOp
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOpYbatch_normalization_11_fusedbatchnormv3_readvariableop_batch_normalization_11_moving_mean*
_output_shapes
:`*
dtype028
6batch_normalization_11/FusedBatchNormV3/ReadVariableOp
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp_batch_normalization_11_fusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance*
_output_shapes
:`*
dtype02:
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_11/FusedBatchNormV3FusedBatchNormV3%depthwise_conv2d_3/depthwise:output:0Dbatch_normalization_11_fusedbatchnormv3_batch_normalization_11_const-batch_normalization_11/ReadVariableOp:value:0>batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:`:`:`:`:`:*
epsilon%o:*
is_training( 2)
'batch_normalization_11/FusedBatchNormV3
re_lu_8/Relu6Relu6+batch_normalization_11/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:`2
re_lu_8/Relu6ท
conv2d_8/Conv2D/ReadVariableOpReadVariableOp.conv2d_8_conv2d_readvariableop_conv2d_8_kernel*&
_output_shapes
:`@*
dtype02 
conv2d_8/Conv2D/ReadVariableOpส
conv2d_8/Conv2DConv2Dre_lu_8/Relu6:activations:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingSAME*
strides
2
conv2d_8/Conv2Dอ
%batch_normalization_12/ReadVariableOpReadVariableOpBbatch_normalization_12_readvariableop_batch_normalization_12_gamma*
_output_shapes
:@*
dtype02'
%batch_normalization_12/ReadVariableOpา
'batch_normalization_12/ReadVariableOp_1ReadVariableOpCbatch_normalization_12_readvariableop_1_batch_normalization_12_beta*
_output_shapes
:@*
dtype02)
'batch_normalization_12/ReadVariableOp_1
6batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOpYbatch_normalization_12_fusedbatchnormv3_readvariableop_batch_normalization_12_moving_mean*
_output_shapes
:@*
dtype028
6batch_normalization_12/FusedBatchNormV3/ReadVariableOp
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp_batch_normalization_12_fusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance*
_output_shapes
:@*
dtype02:
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_12/FusedBatchNormV3FusedBatchNormV3conv2d_8/Conv2D:output:0-batch_normalization_12/ReadVariableOp:value:0/batch_normalization_12/ReadVariableOp_1:value:0>batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:@:@:@:@:@:*
epsilon%o:*
is_training( 2)
'batch_normalization_12/FusedBatchNormV3ฉ
	add_1/addAddV2*batch_normalization_9/FusedBatchNormV3:y:0+batch_normalization_12/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:@2
	add_1/addณ
/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      21
/global_average_pooling2d/Mean/reduction_indicesธ
global_average_pooling2d/MeanMeanadd_1/add:z:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*
_output_shapes

:@2
global_average_pooling2d/Mean
dropout/IdentityIdentity&global_average_pooling2d/Mean:output:0*
T0*
_output_shapes

:@2
dropout/Identityฃ
dense/MatMul/ReadVariableOpReadVariableOp(dense_matmul_readvariableop_dense_kernel*
_output_shapes

:@*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMuldropout/Identity:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp'dense_biasadd_readvariableop_dense_bias*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense/BiasAdda
IdentityIdentitydense/BiasAdd:output:0*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*ม
_input_shapesฏ
ฌ:1(::::: :::::0:::::0::::::::::0:::::0::::::::::0:::::0::::::::::`:::::`::::::::S O
+
_output_shapes
:?????????1(
 
_user_specified_nameinputs: 

_output_shapes
: : 


_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0: (

_output_shapes
:0: -

_output_shapes
:0: 7

_output_shapes
:`: <

_output_shapes
:`
ฤ
u
'__inference_conv2d_3_layer_call_fn_5734

inputs
conv2d_3_kernel
identityขStatefulPartitionedCall์
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_3_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_25782
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*)
_input_shapes
: :22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
: 
 
_user_specified_nameinputs
ี	

5__inference_batch_normalization_12_layer_call_fn_6798

inputs 
batch_normalization_12_gamma
batch_normalization_12_beta&
"batch_normalization_12_moving_mean*
&batch_normalization_12_moving_variance
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_12_gammabatch_normalization_12_beta"batch_normalization_12_moving_mean&batch_normalization_12_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_12_layer_call_and_return_conditional_losses_21562
StatefulPartitionedCallจ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
๗
ใ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1095

inputs-
)readvariableop_batch_normalization_1_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_1_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance0
,fusedbatchnormv3_batch_normalization_1_const
identity
ReadVariableOpReadVariableOp)readvariableop_batch_normalization_1_beta*
_output_shapes
:0*
dtype02
ReadVariableOpภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1๐
FusedBatchNormV3FusedBatchNormV3inputs,fusedbatchnormv3_batch_normalization_1_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:+???????????????????????????0::::0:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs: 

_output_shapes
:0
๗
ใ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1389

inputs-
)readvariableop_batch_normalization_4_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_4_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance0
,fusedbatchnormv3_batch_normalization_4_const
identity
ReadVariableOpReadVariableOp)readvariableop_batch_normalization_4_beta*
_output_shapes
:0*
dtype02
ReadVariableOpภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1๐
FusedBatchNormV3FusedBatchNormV3inputs,fusedbatchnormv3_batch_normalization_4_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:+???????????????????????????0::::0:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs: 

_output_shapes
:0
ง

O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5751

inputs-
)readvariableop_batch_normalization_4_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_4_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance0
,fusedbatchnormv3_batch_normalization_4_const
identityขAssignNewValueขAssignNewValue_1
ReadVariableOpReadVariableOp)readvariableop_batch_normalization_4_beta*
_output_shapes
:0*
dtype02
ReadVariableOpภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ใ
FusedBatchNormV3FusedBatchNormV3inputs,fusedbatchnormv3_batch_normalization_4_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
ื#<2
FusedBatchNormV3ฑ
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_4_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_4/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueว
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_4/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:0::::02 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:N J
&
_output_shapes
:0
 
_user_specified_nameinputs: 

_output_shapes
:0
ษ

B__inference_conv2d_4_layer_call_and_return_conditional_losses_2743

inputs)
%conv2d_readvariableop_conv2d_4_kernel
identity
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_4_kernel*&
_output_shapes
:0 *
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingSAME*
strides
2
Conv2Db
IdentityIdentityConv2D:output:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*)
_input_shapes
:0::N J
&
_output_shapes
:0
 
_user_specified_nameinputs
฿

4__inference_batch_normalization_3_layer_call_fn_5646

inputs
batch_normalization_3_gamma
batch_normalization_3_beta%
!batch_normalization_3_moving_mean)
%batch_normalization_3_moving_variance
identityขStatefulPartitionedCall๏
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_3_gammabatch_normalization_3_beta!batch_normalization_3_moving_mean%batch_normalization_3_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_25142
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": ::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
: 
 
_user_specified_nameinputs
โ
ใ
O__inference_batch_normalization_9_layer_call_and_return_conditional_losses_6427

inputs.
*readvariableop_batch_normalization_9_gamma/
+readvariableop_1_batch_normalization_9_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_9_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_9_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_9_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1ภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_9_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ม
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3g
IdentityIdentityFusedBatchNormV3:y:0*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*5
_input_shapes$
":@:::::N J
&
_output_shapes
:@
 
_user_specified_nameinputs

g
=__inference_add_layer_call_and_return_conditional_losses_2562

inputs
inputs_1
identityV
addAddV2inputsinputs_1*
T0*&
_output_shapes
: 2
addZ
IdentityIdentityadd:z:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*7
_input_shapes&
$: : :N J
&
_output_shapes
: 
 
_user_specified_nameinputs:NJ
&
_output_shapes
: 
 
_user_specified_nameinputs
ฏ
]
A__inference_re_lu_1_layer_call_and_return_conditional_losses_2394

inputs
identityP
Relu6Relu6inputs*
T0*&
_output_shapes
:02
Relu6f
IdentityIdentityRelu6:activations:0*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*%
_input_shapes
:0:N J
&
_output_shapes
:0
 
_user_specified_nameinputs
๋

O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1568

inputs.
*readvariableop_batch_normalization_6_gamma/
+readvariableop_1_batch_normalization_6_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_6_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance
identityขAssignNewValueขAssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_6_gamma*
_output_shapes
: *
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_6_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1ภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_6_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1๊
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o:*
exponential_avg_factor%
ื#<2
FusedBatchNormV3ฑ
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_6_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_6/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueว
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_6/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1ฆ
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
	

4__inference_batch_normalization_8_layer_call_fn_6253

inputs
batch_normalization_8_beta%
!batch_normalization_8_moving_mean)
%batch_normalization_8_moving_variance
batch_normalization_8_const
identityขStatefulPartitionedCall๎
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_8_beta!batch_normalization_8_moving_mean%batch_normalization_8_moving_variancebatch_normalization_8_const*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_29232
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:0::::022
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:0
 
_user_specified_nameinputs: 

_output_shapes
:0
ษ

B__inference_conv2d_4_layer_call_and_return_conditional_losses_5969

inputs)
%conv2d_readvariableop_conv2d_4_kernel
identity
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_4_kernel*&
_output_shapes
:0 *
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingSAME*
strides
2
Conv2Db
IdentityIdentityConv2D:output:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*)
_input_shapes
:0::N J
&
_output_shapes
:0
 
_user_specified_nameinputs
ใ
า)
F__inference_functional_1_layer_call_and_return_conditional_losses_4422
input_1.
*conv2d_conv2d_readvariableop_conv2d_kernel?
;batch_normalization_readvariableop_batch_normalization_betaW
Sbatch_normalization_fusedbatchnormv3_readvariableop_batch_normalization_moving_mean]
Ybatch_normalization_fusedbatchnormv3_readvariableop_1_batch_normalization_moving_varianceB
>batch_normalization_fusedbatchnormv3_batch_normalization_const2
.conv2d_1_conv2d_readvariableop_conv2d_1_kernelC
?batch_normalization_1_readvariableop_batch_normalization_1_beta[
Wbatch_normalization_1_fusedbatchnormv3_readvariableop_batch_normalization_1_moving_meana
]batch_normalization_1_fusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_varianceF
Bbatch_normalization_1_fusedbatchnormv3_batch_normalization_1_constO
Kdepthwise_conv2d_depthwise_readvariableop_depthwise_conv2d_depthwise_kernelC
?batch_normalization_2_readvariableop_batch_normalization_2_beta[
Wbatch_normalization_2_fusedbatchnormv3_readvariableop_batch_normalization_2_moving_meana
]batch_normalization_2_fusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_varianceF
Bbatch_normalization_2_fusedbatchnormv3_batch_normalization_2_const2
.conv2d_2_conv2d_readvariableop_conv2d_2_kernelD
@batch_normalization_3_readvariableop_batch_normalization_3_gammaE
Abatch_normalization_3_readvariableop_1_batch_normalization_3_beta[
Wbatch_normalization_3_fusedbatchnormv3_readvariableop_batch_normalization_3_moving_meana
]batch_normalization_3_fusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance2
.conv2d_3_conv2d_readvariableop_conv2d_3_kernelC
?batch_normalization_4_readvariableop_batch_normalization_4_beta[
Wbatch_normalization_4_fusedbatchnormv3_readvariableop_batch_normalization_4_moving_meana
]batch_normalization_4_fusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_varianceF
Bbatch_normalization_4_fusedbatchnormv3_batch_normalization_4_constS
Odepthwise_conv2d_1_depthwise_readvariableop_depthwise_conv2d_1_depthwise_kernelC
?batch_normalization_5_readvariableop_batch_normalization_5_beta[
Wbatch_normalization_5_fusedbatchnormv3_readvariableop_batch_normalization_5_moving_meana
]batch_normalization_5_fusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_varianceF
Bbatch_normalization_5_fusedbatchnormv3_batch_normalization_5_const2
.conv2d_4_conv2d_readvariableop_conv2d_4_kernelD
@batch_normalization_6_readvariableop_batch_normalization_6_gammaE
Abatch_normalization_6_readvariableop_1_batch_normalization_6_beta[
Wbatch_normalization_6_fusedbatchnormv3_readvariableop_batch_normalization_6_moving_meana
]batch_normalization_6_fusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance2
.conv2d_5_conv2d_readvariableop_conv2d_5_kernelC
?batch_normalization_7_readvariableop_batch_normalization_7_beta[
Wbatch_normalization_7_fusedbatchnormv3_readvariableop_batch_normalization_7_moving_meana
]batch_normalization_7_fusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_varianceF
Bbatch_normalization_7_fusedbatchnormv3_batch_normalization_7_constS
Odepthwise_conv2d_2_depthwise_readvariableop_depthwise_conv2d_2_depthwise_kernelC
?batch_normalization_8_readvariableop_batch_normalization_8_beta[
Wbatch_normalization_8_fusedbatchnormv3_readvariableop_batch_normalization_8_moving_meana
]batch_normalization_8_fusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_varianceF
Bbatch_normalization_8_fusedbatchnormv3_batch_normalization_8_const2
.conv2d_6_conv2d_readvariableop_conv2d_6_kernelD
@batch_normalization_9_readvariableop_batch_normalization_9_gammaE
Abatch_normalization_9_readvariableop_1_batch_normalization_9_beta[
Wbatch_normalization_9_fusedbatchnormv3_readvariableop_batch_normalization_9_moving_meana
]batch_normalization_9_fusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance2
.conv2d_7_conv2d_readvariableop_conv2d_7_kernelE
Abatch_normalization_10_readvariableop_batch_normalization_10_beta]
Ybatch_normalization_10_fusedbatchnormv3_readvariableop_batch_normalization_10_moving_meanc
_batch_normalization_10_fusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_varianceH
Dbatch_normalization_10_fusedbatchnormv3_batch_normalization_10_constS
Odepthwise_conv2d_3_depthwise_readvariableop_depthwise_conv2d_3_depthwise_kernelE
Abatch_normalization_11_readvariableop_batch_normalization_11_beta]
Ybatch_normalization_11_fusedbatchnormv3_readvariableop_batch_normalization_11_moving_meanc
_batch_normalization_11_fusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_varianceH
Dbatch_normalization_11_fusedbatchnormv3_batch_normalization_11_const2
.conv2d_8_conv2d_readvariableop_conv2d_8_kernelF
Bbatch_normalization_12_readvariableop_batch_normalization_12_gammaG
Cbatch_normalization_12_readvariableop_1_batch_normalization_12_beta]
Ybatch_normalization_12_fusedbatchnormv3_readvariableop_batch_normalization_12_moving_meanc
_batch_normalization_12_fusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance,
(dense_matmul_readvariableop_dense_kernel+
'dense_biasadd_readvariableop_dense_bias
identity
%tf_op_layer_ExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%tf_op_layer_ExpandDims/ExpandDims/dimอ
!tf_op_layer_ExpandDims/ExpandDims
ExpandDimsinput_1.tf_op_layer_ExpandDims/ExpandDims/dim:output:0*
T0*
_cloned(*&
_output_shapes
:1(2#
!tf_op_layer_ExpandDims/ExpandDimsฏ
conv2d/Conv2D/ReadVariableOpReadVariableOp*conv2d_conv2d_readvariableop_conv2d_kernel*&
_output_shapes
:( *
dtype02
conv2d/Conv2D/ReadVariableOpิ
conv2d/Conv2DConv2D*tf_op_layer_ExpandDims/ExpandDims:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
2
conv2d/Conv2Dภ
"batch_normalization/ReadVariableOpReadVariableOp;batch_normalization_readvariableop_batch_normalization_beta*
_output_shapes
: *
dtype02$
"batch_normalization/ReadVariableOp๚
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpSbatch_normalization_fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
_output_shapes
: *
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpYbatch_normalization_fusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
_output_shapes
: *
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/Conv2D:output:0>batch_normalization_fusedbatchnormv3_batch_normalization_const*batch_normalization/ReadVariableOp:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
is_training( 2&
$batch_normalization/FusedBatchNormV3~
re_lu/Relu6Relu6(batch_normalization/FusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2
re_lu/Relu6ท
conv2d_1/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_conv2d_readvariableop_conv2d_1_kernel*&
_output_shapes
: 0*
dtype02 
conv2d_1/Conv2D/ReadVariableOpศ
conv2d_1/Conv2DConv2Dre_lu/Relu6:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:0*
paddingSAME*
strides
2
conv2d_1/Conv2Dศ
$batch_normalization_1/ReadVariableOpReadVariableOp?batch_normalization_1_readvariableop_batch_normalization_1_beta*
_output_shapes
:0*
dtype02&
$batch_normalization_1/ReadVariableOp
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_1_fusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
:0*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_1_fusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
_output_shapes
:0*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1๋
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/Conv2D:output:0Bbatch_normalization_1_fusedbatchnormv3_batch_normalization_1_const,batch_normalization_1/ReadVariableOp:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:0:0:0:0:0:*
epsilon%o:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3
re_lu_1/Relu6Relu6*batch_normalization_1/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:02
re_lu_1/Relu6๊
)depthwise_conv2d/depthwise/ReadVariableOpReadVariableOpKdepthwise_conv2d_depthwise_readvariableop_depthwise_conv2d_depthwise_kernel*&
_output_shapes
:0*
dtype02+
)depthwise_conv2d/depthwise/ReadVariableOp
 depthwise_conv2d/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      0      2"
 depthwise_conv2d/depthwise/Shapeฅ
(depthwise_conv2d/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2*
(depthwise_conv2d/depthwise/dilation_rate๚
depthwise_conv2d/depthwiseDepthwiseConv2dNativere_lu_1/Relu6:activations:01depthwise_conv2d/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:0*
paddingSAME*
strides
2
depthwise_conv2d/depthwiseศ
$batch_normalization_2/ReadVariableOpReadVariableOp?batch_normalization_2_readvariableop_batch_normalization_2_beta*
_output_shapes
:0*
dtype02&
$batch_normalization_2/ReadVariableOp
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_2_fusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
:0*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_2_fusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
_output_shapes
:0*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1๖
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3#depthwise_conv2d/depthwise:output:0Bbatch_normalization_2_fusedbatchnormv3_batch_normalization_2_const,batch_normalization_2/ReadVariableOp:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:0:0:0:0:0:*
epsilon%o:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3
re_lu_2/Relu6Relu6*batch_normalization_2/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:02
re_lu_2/Relu6ท
conv2d_2/Conv2D/ReadVariableOpReadVariableOp.conv2d_2_conv2d_readvariableop_conv2d_2_kernel*&
_output_shapes
:0 *
dtype02 
conv2d_2/Conv2D/ReadVariableOpส
conv2d_2/Conv2DConv2Dre_lu_2/Relu6:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingSAME*
strides
2
conv2d_2/Conv2Dษ
$batch_normalization_3/ReadVariableOpReadVariableOp@batch_normalization_3_readvariableop_batch_normalization_3_gamma*
_output_shapes
: *
dtype02&
$batch_normalization_3/ReadVariableOpฮ
&batch_normalization_3/ReadVariableOp_1ReadVariableOpAbatch_normalization_3_readvariableop_1_batch_normalization_3_beta*
_output_shapes
: *
dtype02(
&batch_normalization_3/ReadVariableOp_1
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_3_fusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
_output_shapes
: *
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_3_fusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
_output_shapes
: *
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ื
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_2/Conv2D:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3
add/addAddV2re_lu/Relu6:activations:0*batch_normalization_3/FusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2	
add/addท
conv2d_3/Conv2D/ReadVariableOpReadVariableOp.conv2d_3_conv2d_readvariableop_conv2d_3_kernel*&
_output_shapes
: 0*
dtype02 
conv2d_3/Conv2D/ReadVariableOpบ
conv2d_3/Conv2DConv2Dadd/add:z:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:0*
paddingSAME*
strides
2
conv2d_3/Conv2Dศ
$batch_normalization_4/ReadVariableOpReadVariableOp?batch_normalization_4_readvariableop_batch_normalization_4_beta*
_output_shapes
:0*
dtype02&
$batch_normalization_4/ReadVariableOp
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_4_fusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
_output_shapes
:0*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_4_fusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
_output_shapes
:0*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1๋
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_3/Conv2D:output:0Bbatch_normalization_4_fusedbatchnormv3_batch_normalization_4_const,batch_normalization_4/ReadVariableOp:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:0:0:0:0:0:*
epsilon%o:*
is_training( 2(
&batch_normalization_4/FusedBatchNormV3
re_lu_3/Relu6Relu6*batch_normalization_4/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:02
re_lu_3/Relu6๒
+depthwise_conv2d_1/depthwise/ReadVariableOpReadVariableOpOdepthwise_conv2d_1_depthwise_readvariableop_depthwise_conv2d_1_depthwise_kernel*&
_output_shapes
:0*
dtype02-
+depthwise_conv2d_1/depthwise/ReadVariableOpก
"depthwise_conv2d_1/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      0      2$
"depthwise_conv2d_1/depthwise/Shapeฉ
*depthwise_conv2d_1/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2,
*depthwise_conv2d_1/depthwise/dilation_rate
depthwise_conv2d_1/depthwiseDepthwiseConv2dNativere_lu_3/Relu6:activations:03depthwise_conv2d_1/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:0*
paddingSAME*
strides
2
depthwise_conv2d_1/depthwiseศ
$batch_normalization_5/ReadVariableOpReadVariableOp?batch_normalization_5_readvariableop_batch_normalization_5_beta*
_output_shapes
:0*
dtype02&
$batch_normalization_5/ReadVariableOp
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_5_fusedbatchnormv3_readvariableop_batch_normalization_5_moving_mean*
_output_shapes
:0*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_5_fusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance*
_output_shapes
:0*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1๘
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3%depthwise_conv2d_1/depthwise:output:0Bbatch_normalization_5_fusedbatchnormv3_batch_normalization_5_const,batch_normalization_5/ReadVariableOp:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:0:0:0:0:0:*
epsilon%o:*
is_training( 2(
&batch_normalization_5/FusedBatchNormV3
re_lu_4/Relu6Relu6*batch_normalization_5/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:02
re_lu_4/Relu6ท
conv2d_4/Conv2D/ReadVariableOpReadVariableOp.conv2d_4_conv2d_readvariableop_conv2d_4_kernel*&
_output_shapes
:0 *
dtype02 
conv2d_4/Conv2D/ReadVariableOpส
conv2d_4/Conv2DConv2Dre_lu_4/Relu6:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingSAME*
strides
2
conv2d_4/Conv2Dษ
$batch_normalization_6/ReadVariableOpReadVariableOp@batch_normalization_6_readvariableop_batch_normalization_6_gamma*
_output_shapes
: *
dtype02&
$batch_normalization_6/ReadVariableOpฮ
&batch_normalization_6/ReadVariableOp_1ReadVariableOpAbatch_normalization_6_readvariableop_1_batch_normalization_6_beta*
_output_shapes
: *
dtype02(
&batch_normalization_6/ReadVariableOp_1
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_6_fusedbatchnormv3_readvariableop_batch_normalization_6_moving_mean*
_output_shapes
: *
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_6_fusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance*
_output_shapes
: *
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ื
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv2d_4/Conv2D:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
is_training( 2(
&batch_normalization_6/FusedBatchNormV3ท
conv2d_5/Conv2D/ReadVariableOpReadVariableOp.conv2d_5_conv2d_readvariableop_conv2d_5_kernel*&
_output_shapes
: 0*
dtype02 
conv2d_5/Conv2D/ReadVariableOpู
conv2d_5/Conv2DConv2D*batch_normalization_6/FusedBatchNormV3:y:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:0*
paddingSAME*
strides
2
conv2d_5/Conv2Dศ
$batch_normalization_7/ReadVariableOpReadVariableOp?batch_normalization_7_readvariableop_batch_normalization_7_beta*
_output_shapes
:0*
dtype02&
$batch_normalization_7/ReadVariableOp
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_7_fusedbatchnormv3_readvariableop_batch_normalization_7_moving_mean*
_output_shapes
:0*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_7_fusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance*
_output_shapes
:0*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1๋
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3conv2d_5/Conv2D:output:0Bbatch_normalization_7_fusedbatchnormv3_batch_normalization_7_const,batch_normalization_7/ReadVariableOp:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:0:0:0:0:0:*
epsilon%o:*
is_training( 2(
&batch_normalization_7/FusedBatchNormV3
re_lu_5/Relu6Relu6*batch_normalization_7/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:02
re_lu_5/Relu6๒
+depthwise_conv2d_2/depthwise/ReadVariableOpReadVariableOpOdepthwise_conv2d_2_depthwise_readvariableop_depthwise_conv2d_2_depthwise_kernel*&
_output_shapes
:0*
dtype02-
+depthwise_conv2d_2/depthwise/ReadVariableOpก
"depthwise_conv2d_2/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      0      2$
"depthwise_conv2d_2/depthwise/Shapeฉ
*depthwise_conv2d_2/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2,
*depthwise_conv2d_2/depthwise/dilation_rate
depthwise_conv2d_2/depthwiseDepthwiseConv2dNativere_lu_5/Relu6:activations:03depthwise_conv2d_2/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:0*
paddingSAME*
strides
2
depthwise_conv2d_2/depthwiseศ
$batch_normalization_8/ReadVariableOpReadVariableOp?batch_normalization_8_readvariableop_batch_normalization_8_beta*
_output_shapes
:0*
dtype02&
$batch_normalization_8/ReadVariableOp
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_8_fusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean*
_output_shapes
:0*
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_8_fusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance*
_output_shapes
:0*
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1๘
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3%depthwise_conv2d_2/depthwise:output:0Bbatch_normalization_8_fusedbatchnormv3_batch_normalization_8_const,batch_normalization_8/ReadVariableOp:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:0:0:0:0:0:*
epsilon%o:*
is_training( 2(
&batch_normalization_8/FusedBatchNormV3
re_lu_6/Relu6Relu6*batch_normalization_8/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:02
re_lu_6/Relu6ท
conv2d_6/Conv2D/ReadVariableOpReadVariableOp.conv2d_6_conv2d_readvariableop_conv2d_6_kernel*&
_output_shapes
:0@*
dtype02 
conv2d_6/Conv2D/ReadVariableOpส
conv2d_6/Conv2DConv2Dre_lu_6/Relu6:activations:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingSAME*
strides
2
conv2d_6/Conv2Dษ
$batch_normalization_9/ReadVariableOpReadVariableOp@batch_normalization_9_readvariableop_batch_normalization_9_gamma*
_output_shapes
:@*
dtype02&
$batch_normalization_9/ReadVariableOpฮ
&batch_normalization_9/ReadVariableOp_1ReadVariableOpAbatch_normalization_9_readvariableop_1_batch_normalization_9_beta*
_output_shapes
:@*
dtype02(
&batch_normalization_9/ReadVariableOp_1
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_9_fusedbatchnormv3_readvariableop_batch_normalization_9_moving_mean*
_output_shapes
:@*
dtype027
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_9_fusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance*
_output_shapes
:@*
dtype029
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ื
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3conv2d_6/Conv2D:output:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:@:@:@:@:@:*
epsilon%o:*
is_training( 2(
&batch_normalization_9/FusedBatchNormV3ท
conv2d_7/Conv2D/ReadVariableOpReadVariableOp.conv2d_7_conv2d_readvariableop_conv2d_7_kernel*&
_output_shapes
:@`*
dtype02 
conv2d_7/Conv2D/ReadVariableOpู
conv2d_7/Conv2DConv2D*batch_normalization_9/FusedBatchNormV3:y:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:`*
paddingSAME*
strides
2
conv2d_7/Conv2Dฬ
%batch_normalization_10/ReadVariableOpReadVariableOpAbatch_normalization_10_readvariableop_batch_normalization_10_beta*
_output_shapes
:`*
dtype02'
%batch_normalization_10/ReadVariableOp
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOpYbatch_normalization_10_fusedbatchnormv3_readvariableop_batch_normalization_10_moving_mean*
_output_shapes
:`*
dtype028
6batch_normalization_10/FusedBatchNormV3/ReadVariableOp
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp_batch_normalization_10_fusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance*
_output_shapes
:`*
dtype02:
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1๒
'batch_normalization_10/FusedBatchNormV3FusedBatchNormV3conv2d_7/Conv2D:output:0Dbatch_normalization_10_fusedbatchnormv3_batch_normalization_10_const-batch_normalization_10/ReadVariableOp:value:0>batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:`:`:`:`:`:*
epsilon%o:*
is_training( 2)
'batch_normalization_10/FusedBatchNormV3
re_lu_7/Relu6Relu6+batch_normalization_10/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:`2
re_lu_7/Relu6๒
+depthwise_conv2d_3/depthwise/ReadVariableOpReadVariableOpOdepthwise_conv2d_3_depthwise_readvariableop_depthwise_conv2d_3_depthwise_kernel*&
_output_shapes
:`*
dtype02-
+depthwise_conv2d_3/depthwise/ReadVariableOpก
"depthwise_conv2d_3/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      `      2$
"depthwise_conv2d_3/depthwise/Shapeฉ
*depthwise_conv2d_3/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2,
*depthwise_conv2d_3/depthwise/dilation_rate
depthwise_conv2d_3/depthwiseDepthwiseConv2dNativere_lu_7/Relu6:activations:03depthwise_conv2d_3/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:`*
paddingSAME*
strides
2
depthwise_conv2d_3/depthwiseฬ
%batch_normalization_11/ReadVariableOpReadVariableOpAbatch_normalization_11_readvariableop_batch_normalization_11_beta*
_output_shapes
:`*
dtype02'
%batch_normalization_11/ReadVariableOp
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOpYbatch_normalization_11_fusedbatchnormv3_readvariableop_batch_normalization_11_moving_mean*
_output_shapes
:`*
dtype028
6batch_normalization_11/FusedBatchNormV3/ReadVariableOp
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp_batch_normalization_11_fusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance*
_output_shapes
:`*
dtype02:
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_11/FusedBatchNormV3FusedBatchNormV3%depthwise_conv2d_3/depthwise:output:0Dbatch_normalization_11_fusedbatchnormv3_batch_normalization_11_const-batch_normalization_11/ReadVariableOp:value:0>batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:`:`:`:`:`:*
epsilon%o:*
is_training( 2)
'batch_normalization_11/FusedBatchNormV3
re_lu_8/Relu6Relu6+batch_normalization_11/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:`2
re_lu_8/Relu6ท
conv2d_8/Conv2D/ReadVariableOpReadVariableOp.conv2d_8_conv2d_readvariableop_conv2d_8_kernel*&
_output_shapes
:`@*
dtype02 
conv2d_8/Conv2D/ReadVariableOpส
conv2d_8/Conv2DConv2Dre_lu_8/Relu6:activations:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingSAME*
strides
2
conv2d_8/Conv2Dอ
%batch_normalization_12/ReadVariableOpReadVariableOpBbatch_normalization_12_readvariableop_batch_normalization_12_gamma*
_output_shapes
:@*
dtype02'
%batch_normalization_12/ReadVariableOpา
'batch_normalization_12/ReadVariableOp_1ReadVariableOpCbatch_normalization_12_readvariableop_1_batch_normalization_12_beta*
_output_shapes
:@*
dtype02)
'batch_normalization_12/ReadVariableOp_1
6batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOpYbatch_normalization_12_fusedbatchnormv3_readvariableop_batch_normalization_12_moving_mean*
_output_shapes
:@*
dtype028
6batch_normalization_12/FusedBatchNormV3/ReadVariableOp
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp_batch_normalization_12_fusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance*
_output_shapes
:@*
dtype02:
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_12/FusedBatchNormV3FusedBatchNormV3conv2d_8/Conv2D:output:0-batch_normalization_12/ReadVariableOp:value:0/batch_normalization_12/ReadVariableOp_1:value:0>batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:@:@:@:@:@:*
epsilon%o:*
is_training( 2)
'batch_normalization_12/FusedBatchNormV3ฉ
	add_1/addAddV2*batch_normalization_9/FusedBatchNormV3:y:0+batch_normalization_12/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:@2
	add_1/addณ
/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      21
/global_average_pooling2d/Mean/reduction_indicesธ
global_average_pooling2d/MeanMeanadd_1/add:z:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*
_output_shapes

:@2
global_average_pooling2d/Mean
dropout/IdentityIdentity&global_average_pooling2d/Mean:output:0*
T0*
_output_shapes

:@2
dropout/Identityฃ
dense/MatMul/ReadVariableOpReadVariableOp(dense_matmul_readvariableop_dense_kernel*
_output_shapes

:@*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMuldropout/Identity:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp'dense_biasadd_readvariableop_dense_bias*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense/BiasAdda
IdentityIdentitydense/BiasAdd:output:0*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*ม
_input_shapesฏ
ฌ:1(::::: :::::0:::::0::::::::::0:::::0::::::::::0:::::0::::::::::`:::::`::::::::T P
+
_output_shapes
:?????????1(
!
_user_specified_name	input_1: 

_output_shapes
: : 


_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0: (

_output_shapes
:0: -

_output_shapes
:0: 7

_output_shapes
:`: <

_output_shapes
:`
ษ

B__inference_conv2d_8_layer_call_and_return_conditional_losses_3225

inputs)
%conv2d_readvariableop_conv2d_8_kernel
identity
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_8_kernel*&
_output_shapes
:`@*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingSAME*
strides
2
Conv2Db
IdentityIdentityConv2D:output:0*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*)
_input_shapes
:`::N J
&
_output_shapes
:`
 
_user_specified_nameinputs


O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1477

inputs-
)readvariableop_batch_normalization_5_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_5_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance0
,fusedbatchnormv3_batch_normalization_5_const
identityขAssignNewValueขAssignNewValue_1
ReadVariableOpReadVariableOp)readvariableop_batch_normalization_5_beta*
_output_shapes
:0*
dtype02
ReadVariableOpภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_5_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputs,fusedbatchnormv3_batch_normalization_5_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
ื#<2
FusedBatchNormV3ฑ
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_5_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_5/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueว
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_5/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1ฆ
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:+???????????????????????????0::::02 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs: 

_output_shapes
:0
฿

4__inference_batch_normalization_6_layer_call_fn_6020

inputs
batch_normalization_6_gamma
batch_normalization_6_beta%
!batch_normalization_6_moving_mean)
%batch_normalization_6_moving_variance
identityขStatefulPartitionedCall๏
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_6_gammabatch_normalization_6_beta!batch_normalization_6_moving_mean%batch_normalization_6_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_27702
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": ::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
: 
 
_user_specified_nameinputs
ฤ

@__inference_conv2d_layer_call_and_return_conditional_losses_2234

inputs'
#conv2d_readvariableop_conv2d_kernel
identity
Conv2D/ReadVariableOpReadVariableOp#conv2d_readvariableop_conv2d_kernel*&
_output_shapes
:( *
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
2
Conv2Db
IdentityIdentityConv2D:output:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*)
_input_shapes
:1(::N J
&
_output_shapes
:1(
 
_user_specified_nameinputs

i
=__inference_add_layer_call_and_return_conditional_losses_5715
inputs_0
inputs_1
identityX
addAddV2inputs_0inputs_1*
T0*&
_output_shapes
: 2
addZ
IdentityIdentityadd:z:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*7
_input_shapes&
$: : :P L
&
_output_shapes
: 
"
_user_specified_name
inputs/0:PL
&
_output_shapes
: 
"
_user_specified_name
inputs/1


O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1069

inputs-
)readvariableop_batch_normalization_1_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_1_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance0
,fusedbatchnormv3_batch_normalization_1_const
identityขAssignNewValueขAssignNewValue_1
ReadVariableOpReadVariableOp)readvariableop_batch_normalization_1_beta*
_output_shapes
:0*
dtype02
ReadVariableOpภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputs,fusedbatchnormv3_batch_normalization_1_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
ื#<2
FusedBatchNormV3ฑ
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_1_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_1/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueว
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_1/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1ฆ
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:+???????????????????????????0::::02 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs: 

_output_shapes
:0
?	
๖
2__inference_batch_normalization_layer_call_fn_5285

inputs
batch_normalization_beta#
batch_normalization_moving_mean'
#batch_normalization_moving_variance
batch_normalization_const
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_betabatch_normalization_moving_mean#batch_normalization_moving_variancebatch_normalization_const*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_10072
StatefulPartitionedCallจ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:+??????????????????????????? :::: 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs: 

_output_shapes
: 
?

O__inference_batch_normalization_9_layer_call_and_return_conditional_losses_6409

inputs.
*readvariableop_batch_normalization_9_gamma/
+readvariableop_1_batch_normalization_9_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_9_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance
identityขAssignNewValueขAssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_9_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_9_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1ภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_9_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ฯ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
ื#<2
FusedBatchNormV3ฑ
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_9_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_9/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueว
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_9/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*5
_input_shapes$
":@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:N J
&
_output_shapes
:@
 
_user_specified_nameinputs
ง

O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2604

inputs-
)readvariableop_batch_normalization_4_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_4_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance0
,fusedbatchnormv3_batch_normalization_4_const
identityขAssignNewValueขAssignNewValue_1
ReadVariableOpReadVariableOp)readvariableop_batch_normalization_4_beta*
_output_shapes
:0*
dtype02
ReadVariableOpภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ใ
FusedBatchNormV3FusedBatchNormV3inputs,fusedbatchnormv3_batch_normalization_4_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
ื#<2
FusedBatchNormV3ฑ
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_4_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_4/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueว
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_4/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:0::::02 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:N J
&
_output_shapes
:0
 
_user_specified_nameinputs: 

_output_shapes
:0
อ	

4__inference_batch_normalization_6_layer_call_fn_6083

inputs
batch_normalization_6_gamma
batch_normalization_6_beta%
!batch_normalization_6_moving_mean)
%batch_normalization_6_moving_variance
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_6_gammabatch_normalization_6_beta!batch_normalization_6_moving_mean%batch_normalization_6_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_15952
StatefulPartitionedCallจ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs

ใ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2621

inputs-
)readvariableop_batch_normalization_4_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_4_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance0
,fusedbatchnormv3_batch_normalization_4_const
identity
ReadVariableOpReadVariableOp)readvariableop_batch_normalization_4_beta*
_output_shapes
:0*
dtype02
ReadVariableOpภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ี
FusedBatchNormV3FusedBatchNormV3inputs,fusedbatchnormv3_batch_normalization_4_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:0:0:0:0:0:*
epsilon%o:*
is_training( 2
FusedBatchNormV3g
IdentityIdentityFusedBatchNormV3:y:0*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:0::::0:N J
&
_output_shapes
:0
 
_user_specified_nameinputs: 

_output_shapes
:0
	

4__inference_batch_normalization_1_layer_call_fn_5455

inputs
batch_normalization_1_beta%
!batch_normalization_1_moving_mean)
%batch_normalization_1_moving_variance
batch_normalization_1_const
identityขStatefulPartitionedCall๎
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_1_beta!batch_normalization_1_moving_mean%batch_normalization_1_moving_variancebatch_normalization_1_const*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_23482
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:0::::022
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:0
 
_user_specified_nameinputs: 

_output_shapes
:0
๎	

4__inference_batch_normalization_7_layer_call_fn_6191

inputs
batch_normalization_7_beta%
!batch_normalization_7_moving_mean)
%batch_normalization_7_moving_variance
batch_normalization_7_const
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_7_beta!batch_normalization_7_moving_mean%batch_normalization_7_moving_variancebatch_normalization_7_const*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_16572
StatefulPartitionedCallจ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:+???????????????????????????0::::022
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs: 

_output_shapes
:0
	

4__inference_batch_normalization_2_layer_call_fn_5578

inputs
batch_normalization_2_beta%
!batch_normalization_2_moving_mean)
%batch_normalization_2_moving_variance
batch_normalization_2_const
identityขStatefulPartitionedCall๐
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_2_beta!batch_normalization_2_moving_mean%batch_normalization_2_moving_variancebatch_normalization_2_const*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_24432
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:0::::022
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:0
 
_user_specified_nameinputs: 

_output_shapes
:0
ฏ
]
A__inference_re_lu_1_layer_call_and_return_conditional_losses_5469

inputs
identityP
Relu6Relu6inputs*
T0*&
_output_shapes
:02
Relu6f
IdentityIdentityRelu6:activations:0*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*%
_input_shapes
:0:N J
&
_output_shapes
:0
 
_user_specified_nameinputs
ฏ
]
A__inference_re_lu_6_layer_call_and_return_conditional_losses_6319

inputs
identityP
Relu6Relu6inputs*
T0*&
_output_shapes
:02
Relu6f
IdentityIdentityRelu6:activations:0*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*%
_input_shapes
:0:N J
&
_output_shapes
:0
 
_user_specified_nameinputs
๋

O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1274

inputs.
*readvariableop_batch_normalization_3_gamma/
+readvariableop_1_batch_normalization_3_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_3_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance
identityขAssignNewValueขAssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_3_gamma*
_output_shapes
: *
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_3_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1ภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1๊
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o:*
exponential_avg_factor%
ื#<2
FusedBatchNormV3ฑ
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_3_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_3/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueว
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_3/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1ฆ
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?

O__inference_batch_normalization_9_layer_call_and_return_conditional_losses_3011

inputs.
*readvariableop_batch_normalization_9_gamma/
+readvariableop_1_batch_normalization_9_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_9_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance
identityขAssignNewValueขAssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_9_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_9_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1ภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_9_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ฯ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
ื#<2
FusedBatchNormV3ฑ
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_9_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_9/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueว
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_9/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*5
_input_shapes$
":@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:N J
&
_output_shapes
:@
 
_user_specified_nameinputs
๎	

4__inference_batch_normalization_2_layer_call_fn_5517

inputs
batch_normalization_2_beta%
!batch_normalization_2_moving_mean)
%batch_normalization_2_moving_variance
batch_normalization_2_const
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_2_beta!batch_normalization_2_moving_mean%batch_normalization_2_moving_variancebatch_normalization_2_const*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_11832
StatefulPartitionedCallจ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:+???????????????????????????0::::022
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs: 

_output_shapes
:0

Q
5__inference_tf_op_layer_ExpandDims_layer_call_fn_5220

inputs
identityอ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:1(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_22192
PartitionedCallk
IdentityIdentityPartitionedCall:output:0*
T0*&
_output_shapes
:1(2

Identity"
identityIdentity:output:0*!
_input_shapes
:1(:J F
"
_output_shapes
:1(
 
_user_specified_nameinputs
๙
?
M__inference_batch_normalization_layer_call_and_return_conditional_losses_5250

inputs+
'readvariableop_batch_normalization_betaC
?fusedbatchnormv3_readvariableop_batch_normalization_moving_meanI
Efusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance.
*fusedbatchnormv3_batch_normalization_const
identityขAssignNewValueขAssignNewValue_1
ReadVariableOpReadVariableOp'readvariableop_batch_normalization_beta*
_output_shapes
: *
dtype02
ReadVariableOpพ
FusedBatchNormV3/ReadVariableOpReadVariableOp?fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpศ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpEfusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputs*fusedbatchnormv3_batch_normalization_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o:*
exponential_avg_factor%
ื#<2
FusedBatchNormV3ญ
AssignNewValueAssignVariableOp?fusedbatchnormv3_readvariableop_batch_normalization_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueร
AssignNewValue_1AssignVariableOpEfusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*X
_classN
LJloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1ฆ
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:+??????????????????????????? :::: 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs: 

_output_shapes
: 
ง

O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5429

inputs-
)readvariableop_batch_normalization_1_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_1_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance0
,fusedbatchnormv3_batch_normalization_1_const
identityขAssignNewValueขAssignNewValue_1
ReadVariableOpReadVariableOp)readvariableop_batch_normalization_1_beta*
_output_shapes
:0*
dtype02
ReadVariableOpภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ใ
FusedBatchNormV3FusedBatchNormV3inputs,fusedbatchnormv3_batch_normalization_1_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
ื#<2
FusedBatchNormV3ฑ
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_1_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_1/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueว
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_1/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:0::::02 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:N J
&
_output_shapes
:0
 
_user_specified_nameinputs: 

_output_shapes
:0
ฤ
u
'__inference_conv2d_5_layer_call_fn_6096

inputs
conv2d_5_kernel
identityขStatefulPartitionedCall์
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_5_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_5_layer_call_and_return_conditional_losses_28192
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*)
_input_shapes
: :22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
: 
 
_user_specified_nameinputs
๘

P__inference_batch_normalization_12_layer_call_and_return_conditional_losses_2156

inputs/
+readvariableop_batch_normalization_12_gamma0
,readvariableop_1_batch_normalization_12_betaF
Bfusedbatchnormv3_readvariableop_batch_normalization_12_moving_meanL
Hfusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance
identityขAssignNewValueขAssignNewValue_1
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_12_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_12_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1ม
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_12_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpห
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1๊
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
ื#<2
FusedBatchNormV3ณ
AssignNewValueAssignVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_12_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*U
_classK
IGloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_12/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueษ
AssignNewValue_1AssignVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_12_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*[
_classQ
OMloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_12/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1ฆ
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs

S
7__inference_global_average_pooling2d_layer_call_fn_2209

inputs
identityู
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_22062
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs

B
&__inference_re_lu_4_layer_call_fn_5962

inputs
identityพ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_re_lu_4_layer_call_and_return_conditional_losses_27282
PartitionedCallk
IdentityIdentityPartitionedCall:output:0*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*%
_input_shapes
:0:N J
&
_output_shapes
:0
 
_user_specified_nameinputs
ฏ
]
A__inference_re_lu_6_layer_call_and_return_conditional_losses_2969

inputs
identityP
Relu6Relu6inputs*
T0*&
_output_shapes
:02
Relu6f
IdentityIdentityRelu6:activations:0*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*%
_input_shapes
:0:N J
&
_output_shapes
:0
 
_user_specified_nameinputs
ู	
๖
2__inference_batch_normalization_layer_call_fn_5276

inputs
batch_normalization_beta#
batch_normalization_moving_mean'
#batch_normalization_moving_variance
batch_normalization_const
identityขStatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_betabatch_normalization_moving_mean#batch_normalization_moving_variancebatch_normalization_const*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_batch_normalization_layer_call_and_return_conditional_losses_9812
StatefulPartitionedCallจ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:+??????????????????????????? :::: 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs: 

_output_shapes
: 
ห	

4__inference_batch_normalization_3_layer_call_fn_5700

inputs
batch_normalization_3_gamma
batch_normalization_3_beta%
!batch_normalization_3_moving_mean)
%batch_normalization_3_moving_variance
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_3_gammabatch_normalization_3_beta!batch_normalization_3_moving_mean%batch_normalization_3_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_12742
StatefulPartitionedCallจ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
ญ
[
?__inference_re_lu_layer_call_and_return_conditional_losses_2307

inputs
identityP
Relu6Relu6inputs*
T0*&
_output_shapes
: 2
Relu6f
IdentityIdentityRelu6:activations:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*%
_input_shapes
: :N J
&
_output_shapes
: 
 
_user_specified_nameinputs

i
?__inference_add_1_layer_call_and_return_conditional_losses_3300

inputs
inputs_1
identityV
addAddV2inputsinputs_1*
T0*&
_output_shapes
:@2
addZ
IdentityIdentityadd:z:0*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:@:@:N J
&
_output_shapes
:@
 
_user_specified_nameinputs:NJ
&
_output_shapes
:@
 
_user_specified_nameinputs

ใ
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2940

inputs-
)readvariableop_batch_normalization_8_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_8_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance0
,fusedbatchnormv3_batch_normalization_8_const
identity
ReadVariableOpReadVariableOp)readvariableop_batch_normalization_8_beta*
_output_shapes
:0*
dtype02
ReadVariableOpภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ี
FusedBatchNormV3FusedBatchNormV3inputs,fusedbatchnormv3_batch_normalization_8_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:0:0:0:0:0:*
epsilon%o:*
is_training( 2
FusedBatchNormV3g
IdentityIdentityFusedBatchNormV3:y:0*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:0::::0:N J
&
_output_shapes
:0
 
_user_specified_nameinputs: 

_output_shapes
:0

B
&__inference_re_lu_7_layer_call_fn_6572

inputs
identityพ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_re_lu_7_layer_call_and_return_conditional_losses_31322
PartitionedCallk
IdentityIdentityPartitionedCall:output:0*
T0*&
_output_shapes
:`2

Identity"
identityIdentity:output:0*%
_input_shapes
:`:N J
&
_output_shapes
:`
 
_user_specified_nameinputs
ฤ
u
'__inference_conv2d_1_layer_call_fn_5360

inputs
conv2d_1_kernel
identityขStatefulPartitionedCall์
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_23222
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*)
_input_shapes
: :22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
: 
 
_user_specified_nameinputs
	

5__inference_batch_normalization_10_layer_call_fn_6562

inputs
batch_normalization_10_beta&
"batch_normalization_10_moving_mean*
&batch_normalization_10_moving_variance 
batch_normalization_10_const
identityขStatefulPartitionedCall๕
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_10_beta"batch_normalization_10_moving_mean&batch_normalization_10_moving_variancebatch_normalization_10_const*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:`*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_31032
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:`2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:`::::`22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:`
 
_user_specified_nameinputs: 

_output_shapes
:`
๘	

5__inference_batch_normalization_10_layer_call_fn_6501

inputs
batch_normalization_10_beta&
"batch_normalization_10_moving_mean*
&batch_normalization_10_moving_variance 
batch_normalization_10_const
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_10_beta"batch_normalization_10_moving_mean&batch_normalization_10_moving_variancebatch_normalization_10_const*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????`*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_19512
StatefulPartitionedCallจ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????`2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:+???????????????????????????`::::`22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????`
 
_user_specified_nameinputs: 

_output_shapes
:`


O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1657

inputs-
)readvariableop_batch_normalization_7_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_7_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance0
,fusedbatchnormv3_batch_normalization_7_const
identityขAssignNewValueขAssignNewValue_1
ReadVariableOpReadVariableOp)readvariableop_batch_normalization_7_beta*
_output_shapes
:0*
dtype02
ReadVariableOpภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_7_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputs,fusedbatchnormv3_batch_normalization_7_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
ื#<2
FusedBatchNormV3ฑ
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_7_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_7/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueว
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_7/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1ฆ
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:+???????????????????????????0::::02 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs: 

_output_shapes
:0
ะ

`
A__inference_dropout_layer_call_and_return_conditional_losses_6831

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Constj
dropout/MulMulinputsdropout/Const:output:0*
T0*
_output_shapes

:@2
dropout/Mulo
dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2
dropout/Shapeซ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
_output_shapes

:@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *อฬL>2
dropout/GreaterEqual/yต
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*
_output_shapes

:@2
dropout/GreaterEqualv
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:@2
dropout/Castq
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*
_output_shapes

:@2
dropout/Mul_1\
IdentityIdentitydropout/Mul_1:z:0*
T0*
_output_shapes

:@2

Identity"
identityIdentity:output:0*
_input_shapes

:@:F B

_output_shapes

:@
 
_user_specified_nameinputs
๐	

4__inference_batch_normalization_5_layer_call_fn_5952

inputs
batch_normalization_5_beta%
!batch_normalization_5_moving_mean)
%batch_normalization_5_moving_variance
batch_normalization_5_const
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_5_beta!batch_normalization_5_moving_mean%batch_normalization_5_moving_variancebatch_normalization_5_const*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????0*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_15032
StatefulPartitionedCallจ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:+???????????????????????????0::::022
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs: 

_output_shapes
:0
๗
ใ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1683

inputs-
)readvariableop_batch_normalization_7_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_7_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance0
,fusedbatchnormv3_batch_normalization_7_const
identity
ReadVariableOpReadVariableOp)readvariableop_batch_normalization_7_beta*
_output_shapes
:0*
dtype02
ReadVariableOpภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_7_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1๐
FusedBatchNormV3FusedBatchNormV3inputs,fusedbatchnormv3_batch_normalization_7_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity"
identityIdentity:output:0*R
_input_shapesA
?:+???????????????????????????0::::0:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs: 

_output_shapes
:0

B
&__inference_re_lu_1_layer_call_fn_5474

inputs
identityพ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_re_lu_1_layer_call_and_return_conditional_losses_23942
PartitionedCallk
IdentityIdentityPartitionedCall:output:0*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*%
_input_shapes
:0:N J
&
_output_shapes
:0
 
_user_specified_nameinputs
โ
ใ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_2532

inputs.
*readvariableop_batch_normalization_3_gamma/
+readvariableop_1_batch_normalization_3_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_3_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_3_gamma*
_output_shapes
: *
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_3_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1ภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ม
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3g
IdentityIdentityFusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": :::::N J
&
_output_shapes
: 
 
_user_specified_nameinputs
ษ

B__inference_conv2d_6_layer_call_and_return_conditional_losses_2984

inputs)
%conv2d_readvariableop_conv2d_6_kernel
identity
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_6_kernel*&
_output_shapes
:0@*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingSAME*
strides
2
Conv2Db
IdentityIdentityConv2D:output:0*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*)
_input_shapes
:0::N J
&
_output_shapes
:0
 
_user_specified_nameinputs
	

4__inference_batch_normalization_7_layer_call_fn_6139

inputs
batch_normalization_7_beta%
!batch_normalization_7_moving_mean)
%batch_normalization_7_moving_variance
batch_normalization_7_const
identityขStatefulPartitionedCall๎
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_7_beta!batch_normalization_7_moving_mean%batch_normalization_7_moving_variancebatch_normalization_7_const*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_28452
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:0::::022
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:0
 
_user_specified_nameinputs: 

_output_shapes
:0
ฏ
]
A__inference_re_lu_7_layer_call_and_return_conditional_losses_3132

inputs
identityP
Relu6Relu6inputs*
T0*&
_output_shapes
:`2
Relu6f
IdentityIdentityRelu6:activations:0*
T0*&
_output_shapes
:`2

Identity"
identityIdentity:output:0*%
_input_shapes
:`:N J
&
_output_shapes
:`
 
_user_specified_nameinputs
ำ-
?
+__inference_functional_1_layer_call_fn_5209

inputs
conv2d_kernel
batch_normalization_beta#
batch_normalization_moving_mean'
#batch_normalization_moving_variance
batch_normalization_const
conv2d_1_kernel
batch_normalization_1_beta%
!batch_normalization_1_moving_mean)
%batch_normalization_1_moving_variance
batch_normalization_1_const%
!depthwise_conv2d_depthwise_kernel
batch_normalization_2_beta%
!batch_normalization_2_moving_mean)
%batch_normalization_2_moving_variance
batch_normalization_2_const
conv2d_2_kernel
batch_normalization_3_gamma
batch_normalization_3_beta%
!batch_normalization_3_moving_mean)
%batch_normalization_3_moving_variance
conv2d_3_kernel
batch_normalization_4_beta%
!batch_normalization_4_moving_mean)
%batch_normalization_4_moving_variance
batch_normalization_4_const'
#depthwise_conv2d_1_depthwise_kernel
batch_normalization_5_beta%
!batch_normalization_5_moving_mean)
%batch_normalization_5_moving_variance
batch_normalization_5_const
conv2d_4_kernel
batch_normalization_6_gamma
batch_normalization_6_beta%
!batch_normalization_6_moving_mean)
%batch_normalization_6_moving_variance
conv2d_5_kernel
batch_normalization_7_beta%
!batch_normalization_7_moving_mean)
%batch_normalization_7_moving_variance
batch_normalization_7_const'
#depthwise_conv2d_2_depthwise_kernel
batch_normalization_8_beta%
!batch_normalization_8_moving_mean)
%batch_normalization_8_moving_variance
batch_normalization_8_const
conv2d_6_kernel
batch_normalization_9_gamma
batch_normalization_9_beta%
!batch_normalization_9_moving_mean)
%batch_normalization_9_moving_variance
conv2d_7_kernel
batch_normalization_10_beta&
"batch_normalization_10_moving_mean*
&batch_normalization_10_moving_variance 
batch_normalization_10_const'
#depthwise_conv2d_3_depthwise_kernel
batch_normalization_11_beta&
"batch_normalization_11_moving_mean*
&batch_normalization_11_moving_variance 
batch_normalization_11_const
conv2d_8_kernel 
batch_normalization_12_gamma
batch_normalization_12_beta&
"batch_normalization_12_moving_mean*
&batch_normalization_12_moving_variance
dense_kernel

dense_bias
identityขStatefulPartitionedCallภ
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_kernelbatch_normalization_betabatch_normalization_moving_mean#batch_normalization_moving_variancebatch_normalization_constconv2d_1_kernelbatch_normalization_1_beta!batch_normalization_1_moving_mean%batch_normalization_1_moving_variancebatch_normalization_1_const!depthwise_conv2d_depthwise_kernelbatch_normalization_2_beta!batch_normalization_2_moving_mean%batch_normalization_2_moving_variancebatch_normalization_2_constconv2d_2_kernelbatch_normalization_3_gammabatch_normalization_3_beta!batch_normalization_3_moving_mean%batch_normalization_3_moving_varianceconv2d_3_kernelbatch_normalization_4_beta!batch_normalization_4_moving_mean%batch_normalization_4_moving_variancebatch_normalization_4_const#depthwise_conv2d_1_depthwise_kernelbatch_normalization_5_beta!batch_normalization_5_moving_mean%batch_normalization_5_moving_variancebatch_normalization_5_constconv2d_4_kernelbatch_normalization_6_gammabatch_normalization_6_beta!batch_normalization_6_moving_mean%batch_normalization_6_moving_varianceconv2d_5_kernelbatch_normalization_7_beta!batch_normalization_7_moving_mean%batch_normalization_7_moving_variancebatch_normalization_7_const#depthwise_conv2d_2_depthwise_kernelbatch_normalization_8_beta!batch_normalization_8_moving_mean%batch_normalization_8_moving_variancebatch_normalization_8_constconv2d_6_kernelbatch_normalization_9_gammabatch_normalization_9_beta!batch_normalization_9_moving_mean%batch_normalization_9_moving_varianceconv2d_7_kernelbatch_normalization_10_beta"batch_normalization_10_moving_mean&batch_normalization_10_moving_variancebatch_normalization_10_const#depthwise_conv2d_3_depthwise_kernelbatch_normalization_11_beta"batch_normalization_11_moving_mean&batch_normalization_11_moving_variancebatch_normalization_11_constconv2d_8_kernelbatch_normalization_12_gammabatch_normalization_12_beta"batch_normalization_12_moving_mean&batch_normalization_12_moving_variancedense_kernel
dense_bias*O
TinH
F2D*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*\
_read_only_resource_inputs>
<:	 !"#$%&')*+,./012345689:;=>?@ABC*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_functional_1_layer_call_and_return_conditional_losses_37792
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*ส
_input_shapesธ
ต:?????????1(::::: :::::0:::::0::::::::::0:::::0::::::::::0:::::0::::::::::`:::::`:::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????1(
 
_user_specified_nameinputs: 

_output_shapes
: : 


_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0: (

_output_shapes
:0: -

_output_shapes
:0: 7

_output_shapes
:`: <

_output_shapes
:`
	

4__inference_batch_normalization_2_layer_call_fn_5569

inputs
batch_normalization_2_beta%
!batch_normalization_2_moving_mean)
%batch_normalization_2_moving_variance
batch_normalization_2_const
identityขStatefulPartitionedCall๎
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_2_beta!batch_normalization_2_moving_mean%batch_normalization_2_moving_variancebatch_normalization_2_const*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_24262
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:0::::022
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:0
 
_user_specified_nameinputs: 

_output_shapes
:0
๋

5__inference_batch_normalization_12_layer_call_fn_6753

inputs 
batch_normalization_12_gamma
batch_normalization_12_beta&
"batch_normalization_12_moving_mean*
&batch_normalization_12_moving_variance
identityขStatefulPartitionedCall๖
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_12_gammabatch_normalization_12_beta"batch_normalization_12_moving_mean&batch_normalization_12_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_12_layer_call_and_return_conditional_losses_32702
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*5
_input_shapes$
":@::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:@
 
_user_specified_nameinputs
แ

4__inference_batch_normalization_3_layer_call_fn_5655

inputs
batch_normalization_3_gamma
batch_normalization_3_beta%
!batch_normalization_3_moving_mean)
%batch_normalization_3_moving_variance
identityขStatefulPartitionedCall๑
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_3_gammabatch_normalization_3_beta!batch_normalization_3_moving_mean%batch_normalization_3_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_25322
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": ::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
: 
 
_user_specified_nameinputs

k
?__inference_add_1_layer_call_and_return_conditional_losses_6813
inputs_0
inputs_1
identityX
addAddV2inputs_0inputs_1*
T0*&
_output_shapes
:@2
addZ
IdentityIdentityadd:z:0*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:@:@:P L
&
_output_shapes
:@
"
_user_specified_name
inputs/0:PL
&
_output_shapes
:@
"
_user_specified_name
inputs/1
ง

O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2923

inputs-
)readvariableop_batch_normalization_8_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_8_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance0
,fusedbatchnormv3_batch_normalization_8_const
identityขAssignNewValueขAssignNewValue_1
ReadVariableOpReadVariableOp)readvariableop_batch_normalization_8_beta*
_output_shapes
:0*
dtype02
ReadVariableOpภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ใ
FusedBatchNormV3FusedBatchNormV3inputs,fusedbatchnormv3_batch_normalization_8_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
ื#<2
FusedBatchNormV3ฑ
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_8_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_8/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueว
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_8/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:0::::02 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:N J
&
_output_shapes
:0
 
_user_specified_nameinputs: 

_output_shapes
:0
ฏ
]
A__inference_re_lu_2_layer_call_and_return_conditional_losses_5583

inputs
identityP
Relu6Relu6inputs*
T0*&
_output_shapes
:02
Relu6f
IdentityIdentityRelu6:activations:0*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*%
_input_shapes
:0:N J
&
_output_shapes
:0
 
_user_specified_nameinputs
แ

4__inference_batch_normalization_6_layer_call_fn_6029

inputs
batch_normalization_6_gamma
batch_normalization_6_beta%
!batch_normalization_6_moving_mean)
%batch_normalization_6_moving_variance
identityขStatefulPartitionedCall๑
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_6_gammabatch_normalization_6_beta!batch_normalization_6_moving_mean%batch_normalization_6_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_27882
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": ::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
: 
 
_user_specified_nameinputs
ฤ
u
'__inference_conv2d_6_layer_call_fn_6337

inputs
conv2d_6_kernel
identityขStatefulPartitionedCall์
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_6_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_6_layer_call_and_return_conditional_losses_29842
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*)
_input_shapes
:0:22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:0
 
_user_specified_nameinputs
ษ

B__inference_conv2d_1_layer_call_and_return_conditional_losses_5354

inputs)
%conv2d_readvariableop_conv2d_1_kernel
identity
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_1_kernel*&
_output_shapes
: 0*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:0*
paddingSAME*
strides
2
Conv2Db
IdentityIdentityConv2D:output:0*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*)
_input_shapes
: ::N J
&
_output_shapes
: 
 
_user_specified_nameinputs
	

4__inference_batch_normalization_5_layer_call_fn_5891

inputs
batch_normalization_5_beta%
!batch_normalization_5_moving_mean)
%batch_normalization_5_moving_variance
batch_normalization_5_const
identityขStatefulPartitionedCall๎
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_5_beta!batch_normalization_5_moving_mean%batch_normalization_5_moving_variancebatch_normalization_5_const*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_26822
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:0::::022
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:0
 
_user_specified_nameinputs: 

_output_shapes
:0
?

P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_1951

inputs.
*readvariableop_batch_normalization_10_betaF
Bfusedbatchnormv3_readvariableop_batch_normalization_10_moving_meanL
Hfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance1
-fusedbatchnormv3_batch_normalization_10_const
identityขAssignNewValueขAssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_10_beta*
_output_shapes
:`*
dtype02
ReadVariableOpม
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_10_moving_mean*
_output_shapes
:`*
dtype02!
FusedBatchNormV3/ReadVariableOpห
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance*
_output_shapes
:`*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputs-fusedbatchnormv3_batch_normalization_10_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????`:`:`:`:`:*
epsilon%o:*
exponential_avg_factor%
ื#<2
FusedBatchNormV3ณ
AssignNewValueAssignVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_10_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*U
_classK
IGloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_10/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueษ
AssignNewValue_1AssignVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*[
_classQ
OMloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_10/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1ฆ
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+???????????????????????????`2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:+???????????????????????????`::::`2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+???????????????????????????`
 
_user_specified_nameinputs: 

_output_shapes
:`
๚	

5__inference_batch_normalization_10_layer_call_fn_6510

inputs
batch_normalization_10_beta&
"batch_normalization_10_moving_mean*
&batch_normalization_10_moving_variance 
batch_normalization_10_const
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_10_beta"batch_normalization_10_moving_mean&batch_normalization_10_moving_variancebatch_normalization_10_const*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????`*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_19772
StatefulPartitionedCallจ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????`2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:+???????????????????????????`::::`22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????`
 
_user_specified_nameinputs: 

_output_shapes
:`
ษ

B__inference_conv2d_5_layer_call_and_return_conditional_losses_2819

inputs)
%conv2d_readvariableop_conv2d_5_kernel
identity
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_5_kernel*&
_output_shapes
: 0*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:0*
paddingSAME*
strides
2
Conv2Db
IdentityIdentityConv2D:output:0*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*)
_input_shapes
: ::N J
&
_output_shapes
: 
 
_user_specified_nameinputs
พ็
&
F__inference_functional_1_layer_call_and_return_conditional_losses_3595

inputs
conv2d_conv2d_kernel0
,batch_normalization_batch_normalization_beta7
3batch_normalization_batch_normalization_moving_mean;
7batch_normalization_batch_normalization_moving_variance1
-batch_normalization_batch_normalization_const
conv2d_1_conv2d_1_kernel4
0batch_normalization_1_batch_normalization_1_beta;
7batch_normalization_1_batch_normalization_1_moving_mean?
;batch_normalization_1_batch_normalization_1_moving_variance5
1batch_normalization_1_batch_normalization_1_const6
2depthwise_conv2d_depthwise_conv2d_depthwise_kernel4
0batch_normalization_2_batch_normalization_2_beta;
7batch_normalization_2_batch_normalization_2_moving_mean?
;batch_normalization_2_batch_normalization_2_moving_variance5
1batch_normalization_2_batch_normalization_2_const
conv2d_2_conv2d_2_kernel5
1batch_normalization_3_batch_normalization_3_gamma4
0batch_normalization_3_batch_normalization_3_beta;
7batch_normalization_3_batch_normalization_3_moving_mean?
;batch_normalization_3_batch_normalization_3_moving_variance
conv2d_3_conv2d_3_kernel4
0batch_normalization_4_batch_normalization_4_beta;
7batch_normalization_4_batch_normalization_4_moving_mean?
;batch_normalization_4_batch_normalization_4_moving_variance5
1batch_normalization_4_batch_normalization_4_const:
6depthwise_conv2d_1_depthwise_conv2d_1_depthwise_kernel4
0batch_normalization_5_batch_normalization_5_beta;
7batch_normalization_5_batch_normalization_5_moving_mean?
;batch_normalization_5_batch_normalization_5_moving_variance5
1batch_normalization_5_batch_normalization_5_const
conv2d_4_conv2d_4_kernel5
1batch_normalization_6_batch_normalization_6_gamma4
0batch_normalization_6_batch_normalization_6_beta;
7batch_normalization_6_batch_normalization_6_moving_mean?
;batch_normalization_6_batch_normalization_6_moving_variance
conv2d_5_conv2d_5_kernel4
0batch_normalization_7_batch_normalization_7_beta;
7batch_normalization_7_batch_normalization_7_moving_mean?
;batch_normalization_7_batch_normalization_7_moving_variance5
1batch_normalization_7_batch_normalization_7_const:
6depthwise_conv2d_2_depthwise_conv2d_2_depthwise_kernel4
0batch_normalization_8_batch_normalization_8_beta;
7batch_normalization_8_batch_normalization_8_moving_mean?
;batch_normalization_8_batch_normalization_8_moving_variance5
1batch_normalization_8_batch_normalization_8_const
conv2d_6_conv2d_6_kernel5
1batch_normalization_9_batch_normalization_9_gamma4
0batch_normalization_9_batch_normalization_9_beta;
7batch_normalization_9_batch_normalization_9_moving_mean?
;batch_normalization_9_batch_normalization_9_moving_variance
conv2d_7_conv2d_7_kernel6
2batch_normalization_10_batch_normalization_10_beta=
9batch_normalization_10_batch_normalization_10_moving_meanA
=batch_normalization_10_batch_normalization_10_moving_variance7
3batch_normalization_10_batch_normalization_10_const:
6depthwise_conv2d_3_depthwise_conv2d_3_depthwise_kernel6
2batch_normalization_11_batch_normalization_11_beta=
9batch_normalization_11_batch_normalization_11_moving_meanA
=batch_normalization_11_batch_normalization_11_moving_variance7
3batch_normalization_11_batch_normalization_11_const
conv2d_8_conv2d_8_kernel7
3batch_normalization_12_batch_normalization_12_gamma6
2batch_normalization_12_batch_normalization_12_beta=
9batch_normalization_12_batch_normalization_12_moving_meanA
=batch_normalization_12_batch_normalization_12_moving_variance
dense_dense_kernel
dense_dense_bias
identityข+batch_normalization/StatefulPartitionedCallข-batch_normalization_1/StatefulPartitionedCallข.batch_normalization_10/StatefulPartitionedCallข.batch_normalization_11/StatefulPartitionedCallข.batch_normalization_12/StatefulPartitionedCallข-batch_normalization_2/StatefulPartitionedCallข-batch_normalization_3/StatefulPartitionedCallข-batch_normalization_4/StatefulPartitionedCallข-batch_normalization_5/StatefulPartitionedCallข-batch_normalization_6/StatefulPartitionedCallข-batch_normalization_7/StatefulPartitionedCallข-batch_normalization_8/StatefulPartitionedCallข-batch_normalization_9/StatefulPartitionedCallขconv2d/StatefulPartitionedCallข conv2d_1/StatefulPartitionedCallข conv2d_2/StatefulPartitionedCallข conv2d_3/StatefulPartitionedCallข conv2d_4/StatefulPartitionedCallข conv2d_5/StatefulPartitionedCallข conv2d_6/StatefulPartitionedCallข conv2d_7/StatefulPartitionedCallข conv2d_8/StatefulPartitionedCallขdense/StatefulPartitionedCallข(depthwise_conv2d/StatefulPartitionedCallข*depthwise_conv2d_1/StatefulPartitionedCallข*depthwise_conv2d_2/StatefulPartitionedCallข*depthwise_conv2d_3/StatefulPartitionedCallขdropout/StatefulPartitionedCall๛
&tf_op_layer_ExpandDims/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:1(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_22192(
&tf_op_layer_ExpandDims/PartitionedCallฆ
conv2d/StatefulPartitionedCallStatefulPartitionedCall/tf_op_layer_ExpandDims/PartitionedCall:output:0conv2d_conv2d_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_22342 
conv2d/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0,batch_normalization_batch_normalization_beta3batch_normalization_batch_normalization_moving_mean7batch_normalization_batch_normalization_moving_variance-batch_normalization_batch_normalization_const*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_22612-
+batch_normalization/StatefulPartitionedCall๖
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_re_lu_layer_call_and_return_conditional_losses_23072
re_lu/PartitionedCall
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0conv2d_1_conv2d_1_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_23222"
 conv2d_1/StatefulPartitionedCall
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:00batch_normalization_1_batch_normalization_1_beta7batch_normalization_1_batch_normalization_1_moving_mean;batch_normalization_1_batch_normalization_1_moving_variance1batch_normalization_1_batch_normalization_1_const*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_23482/
-batch_normalization_1/StatefulPartitionedCall?
re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_re_lu_1_layer_call_and_return_conditional_losses_23942
re_lu_1/PartitionedCallำ
(depthwise_conv2d/StatefulPartitionedCallStatefulPartitionedCall re_lu_1/PartitionedCall:output:02depthwise_conv2d_depthwise_conv2d_depthwise_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_depthwise_conv2d_layer_call_and_return_conditional_losses_11242*
(depthwise_conv2d/StatefulPartitionedCall
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall1depthwise_conv2d/StatefulPartitionedCall:output:00batch_normalization_2_batch_normalization_2_beta7batch_normalization_2_batch_normalization_2_moving_mean;batch_normalization_2_batch_normalization_2_moving_variance1batch_normalization_2_batch_normalization_2_const*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_24262/
-batch_normalization_2/StatefulPartitionedCall?
re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_re_lu_2_layer_call_and_return_conditional_losses_24722
re_lu_2/PartitionedCallก
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall re_lu_2/PartitionedCall:output:0conv2d_2_conv2d_2_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_24872"
 conv2d_2/StatefulPartitionedCall
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:01batch_normalization_3_batch_normalization_3_gamma0batch_normalization_3_batch_normalization_3_beta7batch_normalization_3_batch_normalization_3_moving_mean;batch_normalization_3_batch_normalization_3_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_25142/
-batch_normalization_3/StatefulPartitionedCall
add/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:06batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *F
fAR?
=__inference_add_layer_call_and_return_conditional_losses_25622
add/PartitionedCall
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0conv2d_3_conv2d_3_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_25782"
 conv2d_3/StatefulPartitionedCall
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:00batch_normalization_4_batch_normalization_4_beta7batch_normalization_4_batch_normalization_4_moving_mean;batch_normalization_4_batch_normalization_4_moving_variance1batch_normalization_4_batch_normalization_4_const*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_26042/
-batch_normalization_4/StatefulPartitionedCall?
re_lu_3/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_re_lu_3_layer_call_and_return_conditional_losses_26502
re_lu_3/PartitionedCall?
*depthwise_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall re_lu_3/PartitionedCall:output:06depthwise_conv2d_1_depthwise_conv2d_1_depthwise_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_depthwise_conv2d_1_layer_call_and_return_conditional_losses_14182,
*depthwise_conv2d_1/StatefulPartitionedCall
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall3depthwise_conv2d_1/StatefulPartitionedCall:output:00batch_normalization_5_batch_normalization_5_beta7batch_normalization_5_batch_normalization_5_moving_mean;batch_normalization_5_batch_normalization_5_moving_variance1batch_normalization_5_batch_normalization_5_const*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_26822/
-batch_normalization_5/StatefulPartitionedCall?
re_lu_4/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_re_lu_4_layer_call_and_return_conditional_losses_27282
re_lu_4/PartitionedCallก
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:0conv2d_4_conv2d_4_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_27432"
 conv2d_4/StatefulPartitionedCall
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:01batch_normalization_6_batch_normalization_6_gamma0batch_normalization_6_batch_normalization_6_beta7batch_normalization_6_batch_normalization_6_moving_mean;batch_normalization_6_batch_normalization_6_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_27702/
-batch_normalization_6/StatefulPartitionedCallท
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0conv2d_5_conv2d_5_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_5_layer_call_and_return_conditional_losses_28192"
 conv2d_5/StatefulPartitionedCall
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:00batch_normalization_7_batch_normalization_7_beta7batch_normalization_7_batch_normalization_7_moving_mean;batch_normalization_7_batch_normalization_7_moving_variance1batch_normalization_7_batch_normalization_7_const*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_28452/
-batch_normalization_7/StatefulPartitionedCall?
re_lu_5/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_re_lu_5_layer_call_and_return_conditional_losses_28912
re_lu_5/PartitionedCall?
*depthwise_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall re_lu_5/PartitionedCall:output:06depthwise_conv2d_2_depthwise_conv2d_2_depthwise_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_depthwise_conv2d_2_layer_call_and_return_conditional_losses_17122,
*depthwise_conv2d_2/StatefulPartitionedCall
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall3depthwise_conv2d_2/StatefulPartitionedCall:output:00batch_normalization_8_batch_normalization_8_beta7batch_normalization_8_batch_normalization_8_moving_mean;batch_normalization_8_batch_normalization_8_moving_variance1batch_normalization_8_batch_normalization_8_const*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_29232/
-batch_normalization_8/StatefulPartitionedCall?
re_lu_6/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_re_lu_6_layer_call_and_return_conditional_losses_29692
re_lu_6/PartitionedCallก
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall re_lu_6/PartitionedCall:output:0conv2d_6_conv2d_6_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_6_layer_call_and_return_conditional_losses_29842"
 conv2d_6/StatefulPartitionedCall
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:01batch_normalization_9_batch_normalization_9_gamma0batch_normalization_9_batch_normalization_9_beta7batch_normalization_9_batch_normalization_9_moving_mean;batch_normalization_9_batch_normalization_9_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_9_layer_call_and_return_conditional_losses_30112/
-batch_normalization_9/StatefulPartitionedCallท
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0conv2d_7_conv2d_7_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:`*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_7_layer_call_and_return_conditional_losses_30602"
 conv2d_7/StatefulPartitionedCall?
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:02batch_normalization_10_batch_normalization_10_beta9batch_normalization_10_batch_normalization_10_moving_mean=batch_normalization_10_batch_normalization_10_moving_variance3batch_normalization_10_batch_normalization_10_const*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:`*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_308620
.batch_normalization_10/StatefulPartitionedCall?
re_lu_7/PartitionedCallPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_re_lu_7_layer_call_and_return_conditional_losses_31322
re_lu_7/PartitionedCall?
*depthwise_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_7/PartitionedCall:output:06depthwise_conv2d_3_depthwise_conv2d_3_depthwise_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:`*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_depthwise_conv2d_3_layer_call_and_return_conditional_losses_20062,
*depthwise_conv2d_3/StatefulPartitionedCallช
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall3depthwise_conv2d_3/StatefulPartitionedCall:output:02batch_normalization_11_batch_normalization_11_beta9batch_normalization_11_batch_normalization_11_moving_mean=batch_normalization_11_batch_normalization_11_moving_variance3batch_normalization_11_batch_normalization_11_const*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:`*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_316420
.batch_normalization_11/StatefulPartitionedCall?
re_lu_8/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_re_lu_8_layer_call_and_return_conditional_losses_32102
re_lu_8/PartitionedCallก
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall re_lu_8/PartitionedCall:output:0conv2d_8_conv2d_8_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_8_layer_call_and_return_conditional_losses_32252"
 conv2d_8/StatefulPartitionedCallก
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:03batch_normalization_12_batch_normalization_12_gamma2batch_normalization_12_batch_normalization_12_beta9batch_normalization_12_batch_normalization_12_moving_mean=batch_normalization_12_batch_normalization_12_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_12_layer_call_and_return_conditional_losses_325220
.batch_normalization_12/StatefulPartitionedCallฒ
add_1/PartitionedCallPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:07batch_normalization_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_add_1_layer_call_and_return_conditional_losses_33002
add_1/PartitionedCall
(global_average_pooling2d/PartitionedCallPartitionedCalladd_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_22062*
(global_average_pooling2d/PartitionedCall
dropout/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_33272!
dropout/StatefulPartitionedCallฆ
dense/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_dense_kerneldense_dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_33552
dense/StatefulPartitionedCall

IdentityIdentity&dense/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall/^batch_normalization_12/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall^dense/StatefulPartitionedCall)^depthwise_conv2d/StatefulPartitionedCall+^depthwise_conv2d_1/StatefulPartitionedCall+^depthwise_conv2d_2/StatefulPartitionedCall+^depthwise_conv2d_3/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*ม
_input_shapesฏ
ฌ:1(::::: :::::0:::::0::::::::::0:::::0::::::::::0:::::0::::::::::`:::::`:::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2T
(depthwise_conv2d/StatefulPartitionedCall(depthwise_conv2d/StatefulPartitionedCall2X
*depthwise_conv2d_1/StatefulPartitionedCall*depthwise_conv2d_1/StatefulPartitionedCall2X
*depthwise_conv2d_2/StatefulPartitionedCall*depthwise_conv2d_2/StatefulPartitionedCall2X
*depthwise_conv2d_3/StatefulPartitionedCall*depthwise_conv2d_3/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall:S O
+
_output_shapes
:?????????1(
 
_user_specified_nameinputs: 

_output_shapes
: : 


_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0: (

_output_shapes
:0: -

_output_shapes
:0: 7

_output_shapes
:`: <

_output_shapes
:`
?
_
A__inference_dropout_layer_call_and_return_conditional_losses_3332

inputs

identity_1Q
IdentityIdentityinputs*
T0*
_output_shapes

:@2

Identity`

Identity_1IdentityIdentity:output:0*
T0*
_output_shapes

:@2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes

:@:F B

_output_shapes

:@
 
_user_specified_nameinputs

่
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_6544

inputs.
*readvariableop_batch_normalization_10_betaF
Bfusedbatchnormv3_readvariableop_batch_normalization_10_moving_meanL
Hfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance1
-fusedbatchnormv3_batch_normalization_10_const
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_10_beta*
_output_shapes
:`*
dtype02
ReadVariableOpม
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_10_moving_mean*
_output_shapes
:`*
dtype02!
FusedBatchNormV3/ReadVariableOpห
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance*
_output_shapes
:`*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ึ
FusedBatchNormV3FusedBatchNormV3inputs-fusedbatchnormv3_batch_normalization_10_constReadVariableOp:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:`:`:`:`:`:*
epsilon%o:*
is_training( 2
FusedBatchNormV3g
IdentityIdentityFusedBatchNormV3:y:0*
T0*&
_output_shapes
:`2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:`::::`:N J
&
_output_shapes
:`
 
_user_specified_nameinputs: 

_output_shapes
:`
า
l
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_5215

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*
_cloned(*&
_output_shapes
:1(2

ExpandDimsf
IdentityIdentityExpandDims:output:0*
T0*&
_output_shapes
:1(2

Identity"
identityIdentity:output:0*!
_input_shapes
:1(:J F
"
_output_shapes
:1(
 
_user_specified_nameinputs
โ
ใ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_5637

inputs.
*readvariableop_batch_normalization_3_gamma/
+readvariableop_1_batch_normalization_3_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_3_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_3_gamma*
_output_shapes
: *
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_3_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1ภ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpส
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ม
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3g
IdentityIdentityFusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": :::::N J
&
_output_shapes
: 
 
_user_specified_nameinputs"ธL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*
serving_default
6
input_1+
serving_default_input_1:01(0
dense'
StatefulPartitionedCall:0tensorflow/serving/predict:ๅส

พ๊
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer_with_weights-6
layer-11
layer_with_weights-7
layer-12
layer-13
layer_with_weights-8
layer-14
layer_with_weights-9
layer-15
layer-16
layer_with_weights-10
layer-17
layer_with_weights-11
layer-18
layer-19
layer_with_weights-12
layer-20
layer_with_weights-13
layer-21
layer_with_weights-14
layer-22
layer_with_weights-15
layer-23
layer-24
layer_with_weights-16
layer-25
layer_with_weights-17
layer-26
layer-27
layer_with_weights-18
layer-28
layer_with_weights-19
layer-29
layer_with_weights-20
layer-30
 layer_with_weights-21
 layer-31
!layer-32
"layer_with_weights-22
"layer-33
#layer_with_weights-23
#layer-34
$layer-35
%layer_with_weights-24
%layer-36
&layer_with_weights-25
&layer-37
'layer-38
(layer-39
)layer-40
*layer_with_weights-26
*layer-41
+trainable_variables
,regularization_losses
-	variables
.	keras_api
/
signatures
ํ_default_save_signature
๎__call__
+๏&call_and_return_all_conditional_losses"ํ?
_tf_keras_networkะ?{"class_name": "Functional", "name": "functional_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 49, 40]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "ExpandDims", "trainable": true, "dtype": "float32", "node_def": {"name": "ExpandDims", "op": "ExpandDims", "input": ["input_1", "ExpandDims/dim"], "attr": {"T": {"type": "DT_FLOAT"}, "Tdim": {"type": "DT_INT32"}}}, "constants": {"1": 2}}, "name": "tf_op_layer_ExpandDims", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["tf_op_layer_ExpandDims", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 0, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu", "inbound_nodes": [[["batch_normalization", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["re_lu", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 0, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_1", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_1", "inbound_nodes": [[["batch_normalization_1", 0, 0, {}]]]}, {"class_name": "DepthwiseConv2D", "config": {"name": "depthwise_conv2d", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "name": "depthwise_conv2d", "inbound_nodes": [[["re_lu_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 0, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_2", "inbound_nodes": [[["depthwise_conv2d", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_2", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_2", "inbound_nodes": [[["batch_normalization_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["re_lu_2", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_3", "inbound_nodes": [[["conv2d_2", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add", "trainable": true, "dtype": "float32"}, "name": "add", "inbound_nodes": [[["re_lu", 0, 0, {}], ["batch_normalization_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_3", "inbound_nodes": [[["add", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 0, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_4", "inbound_nodes": [[["conv2d_3", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_3", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_3", "inbound_nodes": [[["batch_normalization_4", 0, 0, {}]]]}, {"class_name": "DepthwiseConv2D", "config": {"name": "depthwise_conv2d_1", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "name": "depthwise_conv2d_1", "inbound_nodes": [[["re_lu_3", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 0, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_5", "inbound_nodes": [[["depthwise_conv2d_1", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_4", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_4", "inbound_nodes": [[["batch_normalization_5", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_4", "inbound_nodes": [[["re_lu_4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_6", "inbound_nodes": [[["conv2d_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_5", "inbound_nodes": [[["batch_normalization_6", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 0, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_7", "inbound_nodes": [[["conv2d_5", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_5", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_5", "inbound_nodes": [[["batch_normalization_7", 0, 0, {}]]]}, {"class_name": "DepthwiseConv2D", "config": {"name": "depthwise_conv2d_2", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "name": "depthwise_conv2d_2", "inbound_nodes": [[["re_lu_5", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 0, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_8", "inbound_nodes": [[["depthwise_conv2d_2", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_6", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_6", "inbound_nodes": [[["batch_normalization_8", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_6", "inbound_nodes": [[["re_lu_6", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_9", "inbound_nodes": [[["conv2d_6", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_7", "trainable": true, "dtype": "float32", "filters": 96, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_7", "inbound_nodes": [[["batch_normalization_9", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_10", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 0, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_10", "inbound_nodes": [[["conv2d_7", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_7", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_7", "inbound_nodes": [[["batch_normalization_10", 0, 0, {}]]]}, {"class_name": "DepthwiseConv2D", "config": {"name": "depthwise_conv2d_3", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "name": "depthwise_conv2d_3", "inbound_nodes": [[["re_lu_7", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_11", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 0, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_11", "inbound_nodes": [[["depthwise_conv2d_3", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_8", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_8", "inbound_nodes": [[["batch_normalization_11", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_8", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_8", "inbound_nodes": [[["re_lu_8", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_12", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_12", "inbound_nodes": [[["conv2d_8", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_1", "trainable": true, "dtype": "float32"}, "name": "add_1", "inbound_nodes": [[["batch_normalization_9", 0, 0, {}], ["batch_normalization_12", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "global_average_pooling2d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_average_pooling2d", "inbound_nodes": [[["add_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["global_average_pooling2d", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 14, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["dropout", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 49, 40]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 49, 40]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "ExpandDims", "trainable": true, "dtype": "float32", "node_def": {"name": "ExpandDims", "op": "ExpandDims", "input": ["input_1", "ExpandDims/dim"], "attr": {"T": {"type": "DT_FLOAT"}, "Tdim": {"type": "DT_INT32"}}}, "constants": {"1": 2}}, "name": "tf_op_layer_ExpandDims", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["tf_op_layer_ExpandDims", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 0, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu", "inbound_nodes": [[["batch_normalization", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["re_lu", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 0, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_1", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_1", "inbound_nodes": [[["batch_normalization_1", 0, 0, {}]]]}, {"class_name": "DepthwiseConv2D", "config": {"name": "depthwise_conv2d", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "name": "depthwise_conv2d", "inbound_nodes": [[["re_lu_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 0, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_2", "inbound_nodes": [[["depthwise_conv2d", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_2", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_2", "inbound_nodes": [[["batch_normalization_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["re_lu_2", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_3", "inbound_nodes": [[["conv2d_2", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add", "trainable": true, "dtype": "float32"}, "name": "add", "inbound_nodes": [[["re_lu", 0, 0, {}], ["batch_normalization_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_3", "inbound_nodes": [[["add", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 0, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_4", "inbound_nodes": [[["conv2d_3", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_3", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_3", "inbound_nodes": [[["batch_normalization_4", 0, 0, {}]]]}, {"class_name": "DepthwiseConv2D", "config": {"name": "depthwise_conv2d_1", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "name": "depthwise_conv2d_1", "inbound_nodes": [[["re_lu_3", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 0, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_5", "inbound_nodes": [[["depthwise_conv2d_1", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_4", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_4", "inbound_nodes": [[["batch_normalization_5", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_4", "inbound_nodes": [[["re_lu_4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_6", "inbound_nodes": [[["conv2d_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_5", "inbound_nodes": [[["batch_normalization_6", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 0, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_7", "inbound_nodes": [[["conv2d_5", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_5", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_5", "inbound_nodes": [[["batch_normalization_7", 0, 0, {}]]]}, {"class_name": "DepthwiseConv2D", "config": {"name": "depthwise_conv2d_2", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "name": "depthwise_conv2d_2", "inbound_nodes": [[["re_lu_5", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 0, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_8", "inbound_nodes": [[["depthwise_conv2d_2", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_6", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_6", "inbound_nodes": [[["batch_normalization_8", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_6", "inbound_nodes": [[["re_lu_6", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_9", "inbound_nodes": [[["conv2d_6", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_7", "trainable": true, "dtype": "float32", "filters": 96, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_7", "inbound_nodes": [[["batch_normalization_9", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_10", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 0, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_10", "inbound_nodes": [[["conv2d_7", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_7", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_7", "inbound_nodes": [[["batch_normalization_10", 0, 0, {}]]]}, {"class_name": "DepthwiseConv2D", "config": {"name": "depthwise_conv2d_3", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "name": "depthwise_conv2d_3", "inbound_nodes": [[["re_lu_7", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_11", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 0, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_11", "inbound_nodes": [[["depthwise_conv2d_3", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_8", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_8", "inbound_nodes": [[["batch_normalization_11", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_8", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_8", "inbound_nodes": [[["re_lu_8", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_12", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_12", "inbound_nodes": [[["conv2d_8", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_1", "trainable": true, "dtype": "float32"}, "name": "add_1", "inbound_nodes": [[["batch_normalization_9", 0, 0, {}], ["batch_normalization_12", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "global_average_pooling2d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_average_pooling2d", "inbound_nodes": [[["add_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["global_average_pooling2d", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 14, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["dropout", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense", 0, 0]]}}}
ํ"๊
_tf_keras_input_layerส{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [1, 49, 40]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 49, 40]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}

0regularization_losses
1trainable_variables
2	variables
3	keras_api
๐__call__
+๑&call_and_return_all_conditional_losses"๙
_tf_keras_layer฿{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_ExpandDims", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "ExpandDims", "trainable": true, "dtype": "float32", "node_def": {"name": "ExpandDims", "op": "ExpandDims", "input": ["input_1", "ExpandDims/dim"], "attr": {"T": {"type": "DT_FLOAT"}, "Tdim": {"type": "DT_INT32"}}}, "constants": {"1": 2}}}
	

4kernel
5regularization_losses
6trainable_variables
7	variables
8	keras_api
๒__call__
+๓&call_and_return_all_conditional_losses"?
_tf_keras_layerโ{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 40}}}}
ู
9axis
:beta
;moving_mean
<moving_variance
=regularization_losses
>trainable_variables
?	variables
@	keras_api
๔__call__
+๕&call_and_return_all_conditional_losses"
_tf_keras_layer๔{"class_name": "BatchNormalization", "name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 0, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}}
่
Aregularization_losses
Btrainable_variables
C	variables
D	keras_api
๖__call__
+๗&call_and_return_all_conditional_losses"ื
_tf_keras_layerฝ{"class_name": "ReLU", "name": "re_lu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}}
	

Ekernel
Fregularization_losses
Gtrainable_variables
H	variables
I	keras_api
๘__call__
+๙&call_and_return_all_conditional_losses"?
_tf_keras_layerๅ{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}}
?
Jaxis
Kbeta
Lmoving_mean
Mmoving_variance
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api
๚__call__
+๛&call_and_return_all_conditional_losses"
_tf_keras_layer๘{"class_name": "BatchNormalization", "name": "batch_normalization_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 0, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 48}}}}
์
Rregularization_losses
Strainable_variables
T	variables
U	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layerม{"class_name": "ReLU", "name": "re_lu_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_1", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}}
ะ	
Vdepthwise_kernel
Wregularization_losses
Xtrainable_variables
Y	variables
Z	keras_api
?__call__
+?&call_and_return_all_conditional_losses"ฉ
_tf_keras_layer{"class_name": "DepthwiseConv2D", "name": "depthwise_conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "depthwise_conv2d", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 48}}}}
?
[axis
\beta
]moving_mean
^moving_variance
_regularization_losses
`trainable_variables
a	variables
b	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer๘{"class_name": "BatchNormalization", "name": "batch_normalization_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 0, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 48}}}}
์
cregularization_losses
dtrainable_variables
e	variables
f	keras_api
__call__
+&call_and_return_all_conditional_losses"?
_tf_keras_layerม{"class_name": "ReLU", "name": "re_lu_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_2", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}}
	

gkernel
hregularization_losses
itrainable_variables
j	variables
k	keras_api
__call__
+&call_and_return_all_conditional_losses"?
_tf_keras_layerๅ{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 48}}}}
๋
laxis
	mgamma
nbeta
omoving_mean
pmoving_variance
qregularization_losses
rtrainable_variables
s	variables
t	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer๛{"class_name": "BatchNormalization", "name": "batch_normalization_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}}
จ
uregularization_losses
vtrainable_variables
w	variables
x	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer?{"class_name": "Add", "name": "add", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "add", "trainable": true, "dtype": "float32"}}
	

ykernel
zregularization_losses
{trainable_variables
|	variables
}	keras_api
__call__
+&call_and_return_all_conditional_losses"?
_tf_keras_layerๅ{"class_name": "Conv2D", "name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}}
ใ
~axis
beta
moving_mean
moving_variance
regularization_losses
trainable_variables
	variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer๘{"class_name": "BatchNormalization", "name": "batch_normalization_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 0, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 48}}}}
๐
regularization_losses
trainable_variables
	variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"?
_tf_keras_layerม{"class_name": "ReLU", "name": "re_lu_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_3", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}}
ู	
depthwise_kernel
regularization_losses
trainable_variables
	variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"ญ
_tf_keras_layer{"class_name": "DepthwiseConv2D", "name": "depthwise_conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "depthwise_conv2d_1", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 48}}}}
ๅ
	axis
	beta
moving_mean
moving_variance
regularization_losses
trainable_variables
	variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer๘{"class_name": "BatchNormalization", "name": "batch_normalization_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 0, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 48}}}}
๐
regularization_losses
trainable_variables
	variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"?
_tf_keras_layerม{"class_name": "ReLU", "name": "re_lu_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_4", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}}
ก	
kernel
regularization_losses
trainable_variables
	variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"?
_tf_keras_layerๅ{"class_name": "Conv2D", "name": "conv2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 48}}}}
๔
	?axis

กgamma
	ขbeta
ฃmoving_mean
คmoving_variance
ฅregularization_losses
ฆtrainable_variables
ง	variables
จ	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer๛{"class_name": "BatchNormalization", "name": "batch_normalization_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}}
ก	
ฉkernel
ชregularization_losses
ซtrainable_variables
ฌ	variables
ญ	keras_api
__call__
+&call_and_return_all_conditional_losses"?
_tf_keras_layerๅ{"class_name": "Conv2D", "name": "conv2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}}
ๅ
	ฎaxis
	ฏbeta
ฐmoving_mean
ฑmoving_variance
ฒregularization_losses
ณtrainable_variables
ด	variables
ต	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer๘{"class_name": "BatchNormalization", "name": "batch_normalization_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 0, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 48}}}}
๐
ถregularization_losses
ทtrainable_variables
ธ	variables
น	keras_api
__call__
+&call_and_return_all_conditional_losses"?
_tf_keras_layerม{"class_name": "ReLU", "name": "re_lu_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_5", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}}
ู	
บdepthwise_kernel
ปregularization_losses
ผtrainable_variables
ฝ	variables
พ	keras_api
?__call__
+ก&call_and_return_all_conditional_losses"ญ
_tf_keras_layer{"class_name": "DepthwiseConv2D", "name": "depthwise_conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "depthwise_conv2d_2", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 48}}}}
ๅ
	ฟaxis
	ภbeta
มmoving_mean
ยmoving_variance
รregularization_losses
ฤtrainable_variables
ล	variables
ฦ	keras_api
ข__call__
+ฃ&call_and_return_all_conditional_losses"
_tf_keras_layer๘{"class_name": "BatchNormalization", "name": "batch_normalization_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 0, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 48}}}}
๐
วregularization_losses
ศtrainable_variables
ษ	variables
ส	keras_api
ค__call__
+ฅ&call_and_return_all_conditional_losses"?
_tf_keras_layerม{"class_name": "ReLU", "name": "re_lu_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_6", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}}
ก	
หkernel
ฬregularization_losses
อtrainable_variables
ฮ	variables
ฯ	keras_api
ฆ__call__
+ง&call_and_return_all_conditional_losses"?
_tf_keras_layerๅ{"class_name": "Conv2D", "name": "conv2d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 48}}}}
๔
	ะaxis

ัgamma
	าbeta
ำmoving_mean
ิmoving_variance
ีregularization_losses
ึtrainable_variables
ื	variables
ุ	keras_api
จ__call__
+ฉ&call_and_return_all_conditional_losses"
_tf_keras_layer๛{"class_name": "BatchNormalization", "name": "batch_normalization_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}}
ก	
ูkernel
ฺregularization_losses
?trainable_variables
?	variables
?	keras_api
ช__call__
+ซ&call_and_return_all_conditional_losses"?
_tf_keras_layerๅ{"class_name": "Conv2D", "name": "conv2d_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_7", "trainable": true, "dtype": "float32", "filters": 96, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}}
็
	?axis
	฿beta
เmoving_mean
แmoving_variance
โregularization_losses
ใtrainable_variables
ไ	variables
ๅ	keras_api
ฌ__call__
+ญ&call_and_return_all_conditional_losses"
_tf_keras_layer๚{"class_name": "BatchNormalization", "name": "batch_normalization_10", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_10", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 0, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 96}}}}
๐
ๆregularization_losses
็trainable_variables
่	variables
้	keras_api
ฎ__call__
+ฏ&call_and_return_all_conditional_losses"?
_tf_keras_layerม{"class_name": "ReLU", "name": "re_lu_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_7", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}}
ู	
๊depthwise_kernel
๋regularization_losses
์trainable_variables
ํ	variables
๎	keras_api
ฐ__call__
+ฑ&call_and_return_all_conditional_losses"ญ
_tf_keras_layer{"class_name": "DepthwiseConv2D", "name": "depthwise_conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "depthwise_conv2d_3", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 96}}}}
็
	๏axis
	๐beta
๑moving_mean
๒moving_variance
๓regularization_losses
๔trainable_variables
๕	variables
๖	keras_api
ฒ__call__
+ณ&call_and_return_all_conditional_losses"
_tf_keras_layer๚{"class_name": "BatchNormalization", "name": "batch_normalization_11", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_11", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 0, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 96}}}}
๐
๗regularization_losses
๘trainable_variables
๙	variables
๚	keras_api
ด__call__
+ต&call_and_return_all_conditional_losses"?
_tf_keras_layerม{"class_name": "ReLU", "name": "re_lu_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_8", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}}
ก	
๛kernel
?regularization_losses
?trainable_variables
?	variables
?	keras_api
ถ__call__
+ท&call_and_return_all_conditional_losses"?
_tf_keras_layerๅ{"class_name": "Conv2D", "name": "conv2d_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_8", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 96}}}}
๖
	axis

gamma
	beta
moving_mean
moving_variance
regularization_losses
trainable_variables
	variables
	keras_api
ธ__call__
+น&call_and_return_all_conditional_losses"
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_12", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_12", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}}
ฐ
regularization_losses
trainable_variables
	variables
	keras_api
บ__call__
+ป&call_and_return_all_conditional_losses"
_tf_keras_layer{"class_name": "Add", "name": "add_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "add_1", "trainable": true, "dtype": "float32"}}

regularization_losses
trainable_variables
	variables
	keras_api
ผ__call__
+ฝ&call_and_return_all_conditional_losses"
_tf_keras_layer๊{"class_name": "GlobalAveragePooling2D", "name": "global_average_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "global_average_pooling2d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
็
regularization_losses
trainable_variables
	variables
	keras_api
พ__call__
+ฟ&call_and_return_all_conditional_losses"า
_tf_keras_layerธ{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
ญ
kernel
	bias
regularization_losses
trainable_variables
	variables
	keras_api
ภ__call__
+ม&call_and_return_all_conditional_losses"
_tf_keras_layerๆ{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 14, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}}
ซ
40
:1
E2
K3
V4
\5
g6
m7
n8
y9
10
11
12
13
ก14
ข15
ฉ16
ฏ17
บ18
ภ19
ห20
ั21
า22
ู23
฿24
๊25
๐26
๛27
28
29
30
31"
trackable_list_wrapper
 "
trackable_list_wrapper

40
:1
;2
<3
E4
K5
L6
M7
V8
\9
]10
^11
g12
m13
n14
o15
p16
y17
18
19
20
21
22
23
24
25
ก26
ข27
ฃ28
ค29
ฉ30
ฏ31
ฐ32
ฑ33
บ34
ภ35
ม36
ย37
ห38
ั39
า40
ำ41
ิ42
ู43
฿44
เ45
แ46
๊47
๐48
๑49
๒50
๛51
52
53
54
55
56
57"
trackable_list_wrapper
ำ
+trainable_variables
,regularization_losses
-	variables
non_trainable_variables
layer_metrics
layers
 layer_regularization_losses
metrics
๎__call__
ํ_default_save_signature
+๏&call_and_return_all_conditional_losses
'๏"call_and_return_conditional_losses"
_generic_user_object
-
ยserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ต
0regularization_losses
1trainable_variables
2	variables
?non_trainable_variables
กlayer_metrics
ขlayers
 ฃlayer_regularization_losses
คmetrics
๐__call__
+๑&call_and_return_all_conditional_losses
'๑"call_and_return_conditional_losses"
_generic_user_object
':%( 2conv2d/kernel
 "
trackable_list_wrapper
'
40"
trackable_list_wrapper
'
40"
trackable_list_wrapper
ต
5regularization_losses
6trainable_variables
7	variables
ฅnon_trainable_variables
ฆlayer_metrics
งlayers
 จlayer_regularization_losses
ฉmetrics
๒__call__
+๓&call_and_return_all_conditional_losses
'๓"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
&:$ 2batch_normalization/beta
/:-  (2batch_normalization/moving_mean
3:1  (2#batch_normalization/moving_variance
 "
trackable_list_wrapper
'
:0"
trackable_list_wrapper
5
:0
;1
<2"
trackable_list_wrapper
ต
=regularization_losses
>trainable_variables
?	variables
ชnon_trainable_variables
ซlayer_metrics
ฌlayers
 ญlayer_regularization_losses
ฎmetrics
๔__call__
+๕&call_and_return_all_conditional_losses
'๕"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ต
Aregularization_losses
Btrainable_variables
C	variables
ฏnon_trainable_variables
ฐlayer_metrics
ฑlayers
 ฒlayer_regularization_losses
ณmetrics
๖__call__
+๗&call_and_return_all_conditional_losses
'๗"call_and_return_conditional_losses"
_generic_user_object
):' 02conv2d_1/kernel
 "
trackable_list_wrapper
'
E0"
trackable_list_wrapper
'
E0"
trackable_list_wrapper
ต
Fregularization_losses
Gtrainable_variables
H	variables
ดnon_trainable_variables
ตlayer_metrics
ถlayers
 ทlayer_regularization_losses
ธmetrics
๘__call__
+๙&call_and_return_all_conditional_losses
'๙"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
(:&02batch_normalization_1/beta
1:/0 (2!batch_normalization_1/moving_mean
5:30 (2%batch_normalization_1/moving_variance
 "
trackable_list_wrapper
'
K0"
trackable_list_wrapper
5
K0
L1
M2"
trackable_list_wrapper
ต
Nregularization_losses
Otrainable_variables
P	variables
นnon_trainable_variables
บlayer_metrics
ปlayers
 ผlayer_regularization_losses
ฝmetrics
๚__call__
+๛&call_and_return_all_conditional_losses
'๛"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ต
Rregularization_losses
Strainable_variables
T	variables
พnon_trainable_variables
ฟlayer_metrics
ภlayers
 มlayer_regularization_losses
ยmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
;:902!depthwise_conv2d/depthwise_kernel
 "
trackable_list_wrapper
'
V0"
trackable_list_wrapper
'
V0"
trackable_list_wrapper
ต
Wregularization_losses
Xtrainable_variables
Y	variables
รnon_trainable_variables
ฤlayer_metrics
ลlayers
 ฦlayer_regularization_losses
วmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
(:&02batch_normalization_2/beta
1:/0 (2!batch_normalization_2/moving_mean
5:30 (2%batch_normalization_2/moving_variance
 "
trackable_list_wrapper
'
\0"
trackable_list_wrapper
5
\0
]1
^2"
trackable_list_wrapper
ต
_regularization_losses
`trainable_variables
a	variables
ศnon_trainable_variables
ษlayer_metrics
สlayers
 หlayer_regularization_losses
ฬmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ต
cregularization_losses
dtrainable_variables
e	variables
อnon_trainable_variables
ฮlayer_metrics
ฯlayers
 ะlayer_regularization_losses
ัmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
):'0 2conv2d_2/kernel
 "
trackable_list_wrapper
'
g0"
trackable_list_wrapper
'
g0"
trackable_list_wrapper
ต
hregularization_losses
itrainable_variables
j	variables
าnon_trainable_variables
ำlayer_metrics
ิlayers
 ีlayer_regularization_losses
ึmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):' 2batch_normalization_3/gamma
(:& 2batch_normalization_3/beta
1:/  (2!batch_normalization_3/moving_mean
5:3  (2%batch_normalization_3/moving_variance
 "
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
<
m0
n1
o2
p3"
trackable_list_wrapper
ต
qregularization_losses
rtrainable_variables
s	variables
ืnon_trainable_variables
ุlayer_metrics
ูlayers
 ฺlayer_regularization_losses
?metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ต
uregularization_losses
vtrainable_variables
w	variables
?non_trainable_variables
?layer_metrics
?layers
 ฿layer_regularization_losses
เmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
):' 02conv2d_3/kernel
 "
trackable_list_wrapper
'
y0"
trackable_list_wrapper
'
y0"
trackable_list_wrapper
ต
zregularization_losses
{trainable_variables
|	variables
แnon_trainable_variables
โlayer_metrics
ใlayers
 ไlayer_regularization_losses
ๅmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
(:&02batch_normalization_4/beta
1:/0 (2!batch_normalization_4/moving_mean
5:30 (2%batch_normalization_4/moving_variance
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
7
0
1
2"
trackable_list_wrapper
ธ
regularization_losses
trainable_variables
	variables
ๆnon_trainable_variables
็layer_metrics
่layers
 ้layer_regularization_losses
๊metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ธ
regularization_losses
trainable_variables
	variables
๋non_trainable_variables
์layer_metrics
ํlayers
 ๎layer_regularization_losses
๏metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
=:;02#depthwise_conv2d_1/depthwise_kernel
 "
trackable_list_wrapper
(
0"
trackable_list_wrapper
(
0"
trackable_list_wrapper
ธ
regularization_losses
trainable_variables
	variables
๐non_trainable_variables
๑layer_metrics
๒layers
 ๓layer_regularization_losses
๔metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
(:&02batch_normalization_5/beta
1:/0 (2!batch_normalization_5/moving_mean
5:30 (2%batch_normalization_5/moving_variance
 "
trackable_list_wrapper
(
0"
trackable_list_wrapper
8
0
1
2"
trackable_list_wrapper
ธ
regularization_losses
trainable_variables
	variables
๕non_trainable_variables
๖layer_metrics
๗layers
 ๘layer_regularization_losses
๙metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ธ
regularization_losses
trainable_variables
	variables
๚non_trainable_variables
๛layer_metrics
?layers
 ?layer_regularization_losses
?metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
):'0 2conv2d_4/kernel
 "
trackable_list_wrapper
(
0"
trackable_list_wrapper
(
0"
trackable_list_wrapper
ธ
regularization_losses
trainable_variables
	variables
?non_trainable_variables
layer_metrics
layers
 layer_regularization_losses
metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):' 2batch_normalization_6/gamma
(:& 2batch_normalization_6/beta
1:/  (2!batch_normalization_6/moving_mean
5:3  (2%batch_normalization_6/moving_variance
 "
trackable_list_wrapper
0
ก0
ข1"
trackable_list_wrapper
@
ก0
ข1
ฃ2
ค3"
trackable_list_wrapper
ธ
ฅregularization_losses
ฆtrainable_variables
ง	variables
non_trainable_variables
layer_metrics
layers
 layer_regularization_losses
metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
):' 02conv2d_5/kernel
 "
trackable_list_wrapper
(
ฉ0"
trackable_list_wrapper
(
ฉ0"
trackable_list_wrapper
ธ
ชregularization_losses
ซtrainable_variables
ฌ	variables
non_trainable_variables
layer_metrics
layers
 layer_regularization_losses
metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
(:&02batch_normalization_7/beta
1:/0 (2!batch_normalization_7/moving_mean
5:30 (2%batch_normalization_7/moving_variance
 "
trackable_list_wrapper
(
ฏ0"
trackable_list_wrapper
8
ฏ0
ฐ1
ฑ2"
trackable_list_wrapper
ธ
ฒregularization_losses
ณtrainable_variables
ด	variables
non_trainable_variables
layer_metrics
layers
 layer_regularization_losses
metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ธ
ถregularization_losses
ทtrainable_variables
ธ	variables
non_trainable_variables
layer_metrics
layers
 layer_regularization_losses
metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
=:;02#depthwise_conv2d_2/depthwise_kernel
 "
trackable_list_wrapper
(
บ0"
trackable_list_wrapper
(
บ0"
trackable_list_wrapper
ธ
ปregularization_losses
ผtrainable_variables
ฝ	variables
non_trainable_variables
layer_metrics
layers
 layer_regularization_losses
metrics
?__call__
+ก&call_and_return_all_conditional_losses
'ก"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
(:&02batch_normalization_8/beta
1:/0 (2!batch_normalization_8/moving_mean
5:30 (2%batch_normalization_8/moving_variance
 "
trackable_list_wrapper
(
ภ0"
trackable_list_wrapper
8
ภ0
ม1
ย2"
trackable_list_wrapper
ธ
รregularization_losses
ฤtrainable_variables
ล	variables
non_trainable_variables
layer_metrics
layers
 ?layer_regularization_losses
กmetrics
ข__call__
+ฃ&call_and_return_all_conditional_losses
'ฃ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ธ
วregularization_losses
ศtrainable_variables
ษ	variables
ขnon_trainable_variables
ฃlayer_metrics
คlayers
 ฅlayer_regularization_losses
ฆmetrics
ค__call__
+ฅ&call_and_return_all_conditional_losses
'ฅ"call_and_return_conditional_losses"
_generic_user_object
):'0@2conv2d_6/kernel
 "
trackable_list_wrapper
(
ห0"
trackable_list_wrapper
(
ห0"
trackable_list_wrapper
ธ
ฬregularization_losses
อtrainable_variables
ฮ	variables
งnon_trainable_variables
จlayer_metrics
ฉlayers
 ชlayer_regularization_losses
ซmetrics
ฆ__call__
+ง&call_and_return_all_conditional_losses
'ง"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2batch_normalization_9/gamma
(:&@2batch_normalization_9/beta
1:/@ (2!batch_normalization_9/moving_mean
5:3@ (2%batch_normalization_9/moving_variance
 "
trackable_list_wrapper
0
ั0
า1"
trackable_list_wrapper
@
ั0
า1
ำ2
ิ3"
trackable_list_wrapper
ธ
ีregularization_losses
ึtrainable_variables
ื	variables
ฌnon_trainable_variables
ญlayer_metrics
ฎlayers
 ฏlayer_regularization_losses
ฐmetrics
จ__call__
+ฉ&call_and_return_all_conditional_losses
'ฉ"call_and_return_conditional_losses"
_generic_user_object
):'@`2conv2d_7/kernel
 "
trackable_list_wrapper
(
ู0"
trackable_list_wrapper
(
ู0"
trackable_list_wrapper
ธ
ฺregularization_losses
?trainable_variables
?	variables
ฑnon_trainable_variables
ฒlayer_metrics
ณlayers
 ดlayer_regularization_losses
ตmetrics
ช__call__
+ซ&call_and_return_all_conditional_losses
'ซ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'`2batch_normalization_10/beta
2:0` (2"batch_normalization_10/moving_mean
6:4` (2&batch_normalization_10/moving_variance
 "
trackable_list_wrapper
(
฿0"
trackable_list_wrapper
8
฿0
เ1
แ2"
trackable_list_wrapper
ธ
โregularization_losses
ใtrainable_variables
ไ	variables
ถnon_trainable_variables
ทlayer_metrics
ธlayers
 นlayer_regularization_losses
บmetrics
ฌ__call__
+ญ&call_and_return_all_conditional_losses
'ญ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ธ
ๆregularization_losses
็trainable_variables
่	variables
ปnon_trainable_variables
ผlayer_metrics
ฝlayers
 พlayer_regularization_losses
ฟmetrics
ฎ__call__
+ฏ&call_and_return_all_conditional_losses
'ฏ"call_and_return_conditional_losses"
_generic_user_object
=:;`2#depthwise_conv2d_3/depthwise_kernel
 "
trackable_list_wrapper
(
๊0"
trackable_list_wrapper
(
๊0"
trackable_list_wrapper
ธ
๋regularization_losses
์trainable_variables
ํ	variables
ภnon_trainable_variables
มlayer_metrics
ยlayers
 รlayer_regularization_losses
ฤmetrics
ฐ__call__
+ฑ&call_and_return_all_conditional_losses
'ฑ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'`2batch_normalization_11/beta
2:0` (2"batch_normalization_11/moving_mean
6:4` (2&batch_normalization_11/moving_variance
 "
trackable_list_wrapper
(
๐0"
trackable_list_wrapper
8
๐0
๑1
๒2"
trackable_list_wrapper
ธ
๓regularization_losses
๔trainable_variables
๕	variables
ลnon_trainable_variables
ฦlayer_metrics
วlayers
 ศlayer_regularization_losses
ษmetrics
ฒ__call__
+ณ&call_and_return_all_conditional_losses
'ณ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ธ
๗regularization_losses
๘trainable_variables
๙	variables
สnon_trainable_variables
หlayer_metrics
ฬlayers
 อlayer_regularization_losses
ฮmetrics
ด__call__
+ต&call_and_return_all_conditional_losses
'ต"call_and_return_conditional_losses"
_generic_user_object
):'`@2conv2d_8/kernel
 "
trackable_list_wrapper
(
๛0"
trackable_list_wrapper
(
๛0"
trackable_list_wrapper
ธ
?regularization_losses
?trainable_variables
?	variables
ฯnon_trainable_variables
ะlayer_metrics
ัlayers
 าlayer_regularization_losses
ำmetrics
ถ__call__
+ท&call_and_return_all_conditional_losses
'ท"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(@2batch_normalization_12/gamma
):'@2batch_normalization_12/beta
2:0@ (2"batch_normalization_12/moving_mean
6:4@ (2&batch_normalization_12/moving_variance
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
@
0
1
2
3"
trackable_list_wrapper
ธ
regularization_losses
trainable_variables
	variables
ิnon_trainable_variables
ีlayer_metrics
ึlayers
 ืlayer_regularization_losses
ุmetrics
ธ__call__
+น&call_and_return_all_conditional_losses
'น"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ธ
regularization_losses
trainable_variables
	variables
ูnon_trainable_variables
ฺlayer_metrics
?layers
 ?layer_regularization_losses
?metrics
บ__call__
+ป&call_and_return_all_conditional_losses
'ป"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ธ
regularization_losses
trainable_variables
	variables
?non_trainable_variables
฿layer_metrics
เlayers
 แlayer_regularization_losses
โmetrics
ผ__call__
+ฝ&call_and_return_all_conditional_losses
'ฝ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ธ
regularization_losses
trainable_variables
	variables
ใnon_trainable_variables
ไlayer_metrics
ๅlayers
 ๆlayer_regularization_losses
็metrics
พ__call__
+ฟ&call_and_return_all_conditional_losses
'ฟ"call_and_return_conditional_losses"
_generic_user_object
:@2dense/kernel
:2
dense/bias
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
ธ
regularization_losses
trainable_variables
	variables
่non_trainable_variables
้layer_metrics
๊layers
 ๋layer_regularization_losses
์metrics
ภ__call__
+ม&call_and_return_all_conditional_losses
'ม"call_and_return_conditional_losses"
_generic_user_object
๘
;0
<1
L2
M3
]4
^5
o6
p7
8
9
10
11
ฃ12
ค13
ฐ14
ฑ15
ม16
ย17
ำ18
ิ19
เ20
แ21
๑22
๒23
24
25"
trackable_list_wrapper
 "
trackable_dict_wrapper
ๆ
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
o0
p1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
ฃ0
ค1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
ฐ0
ฑ1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
ม0
ย1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
ำ0
ิ1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
เ0
แ1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
๑0
๒1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
เ2?
__inference__wrapped_model_926บ
ฒ
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช **ข'
%"
input_1?????????1(
๚2๗
+__inference_functional_1_layer_call_fn_4494
+__inference_functional_1_layer_call_fn_4566
+__inference_functional_1_layer_call_fn_5137
+__inference_functional_1_layer_call_fn_5209ภ
ทฒณ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
ๆ2ใ
F__inference_functional_1_layer_call_and_return_conditional_losses_4422
F__inference_functional_1_layer_call_and_return_conditional_losses_4819
F__inference_functional_1_layer_call_and_return_conditional_losses_5065
F__inference_functional_1_layer_call_and_return_conditional_losses_4176ภ
ทฒณ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
฿2?
5__inference_tf_op_layer_ExpandDims_layer_call_fn_5220ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
๚2๗
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_5215ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
ฯ2ฬ
%__inference_conv2d_layer_call_fn_5233ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
๊2็
@__inference_conv2d_layer_call_and_return_conditional_losses_5227ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
2
2__inference_batch_normalization_layer_call_fn_5328
2__inference_batch_normalization_layer_call_fn_5276
2__inference_batch_normalization_layer_call_fn_5337
2__inference_batch_normalization_layer_call_fn_5285ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
๖2๓
M__inference_batch_normalization_layer_call_and_return_conditional_losses_5319
M__inference_batch_normalization_layer_call_and_return_conditional_losses_5267
M__inference_batch_normalization_layer_call_and_return_conditional_losses_5302
M__inference_batch_normalization_layer_call_and_return_conditional_losses_5250ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
ฮ2ห
$__inference_re_lu_layer_call_fn_5347ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
้2ๆ
?__inference_re_lu_layer_call_and_return_conditional_losses_5342ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
ั2ฮ
'__inference_conv2d_1_layer_call_fn_5360ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
์2้
B__inference_conv2d_1_layer_call_and_return_conditional_losses_5354ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
2
4__inference_batch_normalization_1_layer_call_fn_5403
4__inference_batch_normalization_1_layer_call_fn_5412
4__inference_batch_normalization_1_layer_call_fn_5455
4__inference_batch_normalization_1_layer_call_fn_5464ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
?2๛
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5429
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5377
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5394
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5446ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
ะ2อ
&__inference_re_lu_1_layer_call_fn_5474ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
๋2่
A__inference_re_lu_1_layer_call_and_return_conditional_losses_5469ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
2
/__inference_depthwise_conv2d_layer_call_fn_1128ื
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *7ข4
2/+???????????????????????????0
ฉ2ฆ
J__inference_depthwise_conv2d_layer_call_and_return_conditional_losses_1112ื
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *7ข4
2/+???????????????????????????0
2
4__inference_batch_normalization_2_layer_call_fn_5517
4__inference_batch_normalization_2_layer_call_fn_5569
4__inference_batch_normalization_2_layer_call_fn_5526
4__inference_batch_normalization_2_layer_call_fn_5578ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
?2๛
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5543
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5560
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5508
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5491ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
ะ2อ
&__inference_re_lu_2_layer_call_fn_5588ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
๋2่
A__inference_re_lu_2_layer_call_and_return_conditional_losses_5583ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
ั2ฮ
'__inference_conv2d_2_layer_call_fn_5601ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
์2้
B__inference_conv2d_2_layer_call_and_return_conditional_losses_5595ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
2
4__inference_batch_normalization_3_layer_call_fn_5709
4__inference_batch_normalization_3_layer_call_fn_5700
4__inference_batch_normalization_3_layer_call_fn_5655
4__inference_batch_normalization_3_layer_call_fn_5646ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
?2๛
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_5691
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_5619
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_5637
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_5673ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
ฬ2ษ
"__inference_add_layer_call_fn_5721ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
็2ไ
=__inference_add_layer_call_and_return_conditional_losses_5715ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
ั2ฮ
'__inference_conv2d_3_layer_call_fn_5734ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
์2้
B__inference_conv2d_3_layer_call_and_return_conditional_losses_5728ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
2
4__inference_batch_normalization_4_layer_call_fn_5786
4__inference_batch_normalization_4_layer_call_fn_5777
4__inference_batch_normalization_4_layer_call_fn_5829
4__inference_batch_normalization_4_layer_call_fn_5838ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
?2๛
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5820
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5768
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5751
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5803ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
ะ2อ
&__inference_re_lu_3_layer_call_fn_5848ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
๋2่
A__inference_re_lu_3_layer_call_and_return_conditional_losses_5843ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
2
1__inference_depthwise_conv2d_1_layer_call_fn_1422ื
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *7ข4
2/+???????????????????????????0
ซ2จ
L__inference_depthwise_conv2d_1_layer_call_and_return_conditional_losses_1406ื
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *7ข4
2/+???????????????????????????0
2
4__inference_batch_normalization_5_layer_call_fn_5943
4__inference_batch_normalization_5_layer_call_fn_5952
4__inference_batch_normalization_5_layer_call_fn_5891
4__inference_batch_normalization_5_layer_call_fn_5900ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
?2๛
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_5917
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_5934
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_5865
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_5882ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
ะ2อ
&__inference_re_lu_4_layer_call_fn_5962ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
๋2่
A__inference_re_lu_4_layer_call_and_return_conditional_losses_5957ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
ั2ฮ
'__inference_conv2d_4_layer_call_fn_5975ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
์2้
B__inference_conv2d_4_layer_call_and_return_conditional_losses_5969ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
2
4__inference_batch_normalization_6_layer_call_fn_6074
4__inference_batch_normalization_6_layer_call_fn_6029
4__inference_batch_normalization_6_layer_call_fn_6020
4__inference_batch_normalization_6_layer_call_fn_6083ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
?2๛
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_6047
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_6011
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_5993
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_6065ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
ั2ฮ
'__inference_conv2d_5_layer_call_fn_6096ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
์2้
B__inference_conv2d_5_layer_call_and_return_conditional_losses_6090ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
2
4__inference_batch_normalization_7_layer_call_fn_6200
4__inference_batch_normalization_7_layer_call_fn_6148
4__inference_batch_normalization_7_layer_call_fn_6139
4__inference_batch_normalization_7_layer_call_fn_6191ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
?2๛
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_6130
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_6182
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_6113
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_6165ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
ะ2อ
&__inference_re_lu_5_layer_call_fn_6210ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
๋2่
A__inference_re_lu_5_layer_call_and_return_conditional_losses_6205ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
2
1__inference_depthwise_conv2d_2_layer_call_fn_1716ื
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *7ข4
2/+???????????????????????????0
ซ2จ
L__inference_depthwise_conv2d_2_layer_call_and_return_conditional_losses_1700ื
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *7ข4
2/+???????????????????????????0
2
4__inference_batch_normalization_8_layer_call_fn_6305
4__inference_batch_normalization_8_layer_call_fn_6314
4__inference_batch_normalization_8_layer_call_fn_6253
4__inference_batch_normalization_8_layer_call_fn_6262ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
?2๛
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_6244
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_6227
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_6296
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_6279ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
ะ2อ
&__inference_re_lu_6_layer_call_fn_6324ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
๋2่
A__inference_re_lu_6_layer_call_and_return_conditional_losses_6319ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
ั2ฮ
'__inference_conv2d_6_layer_call_fn_6337ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
์2้
B__inference_conv2d_6_layer_call_and_return_conditional_losses_6331ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
2
4__inference_batch_normalization_9_layer_call_fn_6391
4__inference_batch_normalization_9_layer_call_fn_6445
4__inference_batch_normalization_9_layer_call_fn_6436
4__inference_batch_normalization_9_layer_call_fn_6382ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
?2๛
O__inference_batch_normalization_9_layer_call_and_return_conditional_losses_6355
O__inference_batch_normalization_9_layer_call_and_return_conditional_losses_6373
O__inference_batch_normalization_9_layer_call_and_return_conditional_losses_6427
O__inference_batch_normalization_9_layer_call_and_return_conditional_losses_6409ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
ั2ฮ
'__inference_conv2d_7_layer_call_fn_6458ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
์2้
B__inference_conv2d_7_layer_call_and_return_conditional_losses_6452ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
2
5__inference_batch_normalization_10_layer_call_fn_6562
5__inference_batch_normalization_10_layer_call_fn_6510
5__inference_batch_normalization_10_layer_call_fn_6501
5__inference_batch_normalization_10_layer_call_fn_6553ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
2?
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_6527
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_6475
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_6492
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_6544ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
ะ2อ
&__inference_re_lu_7_layer_call_fn_6572ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
๋2่
A__inference_re_lu_7_layer_call_and_return_conditional_losses_6567ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
2
1__inference_depthwise_conv2d_3_layer_call_fn_2010ื
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *7ข4
2/+???????????????????????????`
ซ2จ
L__inference_depthwise_conv2d_3_layer_call_and_return_conditional_losses_1994ื
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *7ข4
2/+???????????????????????????`
2
5__inference_batch_normalization_11_layer_call_fn_6615
5__inference_batch_normalization_11_layer_call_fn_6676
5__inference_batch_normalization_11_layer_call_fn_6667
5__inference_batch_normalization_11_layer_call_fn_6624ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
2?
P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_6641
P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_6606
P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_6589
P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_6658ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
ะ2อ
&__inference_re_lu_8_layer_call_fn_6686ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
๋2่
A__inference_re_lu_8_layer_call_and_return_conditional_losses_6681ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
ั2ฮ
'__inference_conv2d_8_layer_call_fn_6699ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
์2้
B__inference_conv2d_8_layer_call_and_return_conditional_losses_6693ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
2
5__inference_batch_normalization_12_layer_call_fn_6807
5__inference_batch_normalization_12_layer_call_fn_6744
5__inference_batch_normalization_12_layer_call_fn_6798
5__inference_batch_normalization_12_layer_call_fn_6753ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
2?
P__inference_batch_normalization_12_layer_call_and_return_conditional_losses_6771
P__inference_batch_normalization_12_layer_call_and_return_conditional_losses_6789
P__inference_batch_normalization_12_layer_call_and_return_conditional_losses_6717
P__inference_batch_normalization_12_layer_call_and_return_conditional_losses_6735ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
ฮ2ห
$__inference_add_1_layer_call_fn_6819ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
้2ๆ
?__inference_add_1_layer_call_and_return_conditional_losses_6813ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
2
7__inference_global_average_pooling2d_layer_call_fn_2209เ
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *@ข=
;84????????????????????????????????????
บ2ท
R__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_2197เ
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *@ข=
;84????????????????????????????????????
2
&__inference_dropout_layer_call_fn_6846
&__inference_dropout_layer_call_fn_6841ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
ภ2ฝ
A__inference_dropout_layer_call_and_return_conditional_losses_6831
A__inference_dropout_layer_call_and_return_conditional_losses_6836ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
ฮ2ห
$__inference_dense_layer_call_fn_6863ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
้2ๆ
?__inference_dense_layer_call_and_return_conditional_losses_6856ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
1B/
"__inference_signature_wrapper_3923input_1
	J
Const
J	
Const_1
J	
Const_2
J	
Const_3
J	
Const_4
J	
Const_5
J	
Const_6
J	
Const_7
J	
Const_8๔
__inference__wrapped_model_926ัs4:;<รEKLMฤV\]^ลgmnopyฦวกขฃคฉฏฐฑศบภมยษหัาำิู฿เแส๊๐๑๒ห๛4ข1
*ข'
%"
input_1?????????1(
ช "$ช!

dense
denseฤ
?__inference_add_1_layer_call_and_return_conditional_losses_6813XขU
NขK
IF
!
inputs/0@
!
inputs/1@
ช "$ข!

0@
 
$__inference_add_1_layer_call_fn_6819sXขU
NขK
IF
!
inputs/0@
!
inputs/1@
ช "@ย
=__inference_add_layer_call_and_return_conditional_losses_5715XขU
NขK
IF
!
inputs/0 
!
inputs/1 
ช "$ข!

0 
 
"__inference_add_layer_call_fn_5721sXขU
NขK
IF
!
inputs/0 
!
inputs/1 
ช " ๏
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_6475฿เแสMขJ
Cข@
:7
inputs+???????????????????????????`
p
ช "?ข<
52
0+???????????????????????????`
 ๏
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_6492฿เแสMขJ
Cข@
:7
inputs+???????????????????????????`
p 
ช "?ข<
52
0+???????????????????????????`
 ธ
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_6527d฿เแส2ข/
(ข%

inputs`
p
ช "$ข!

0`
 ธ
P__inference_batch_normalization_10_layer_call_and_return_conditional_losses_6544d฿เแส2ข/
(ข%

inputs`
p 
ช "$ข!

0`
 ว
5__inference_batch_normalization_10_layer_call_fn_6501฿เแสMขJ
Cข@
:7
inputs+???????????????????????????`
p
ช "2/+???????????????????????????`ว
5__inference_batch_normalization_10_layer_call_fn_6510฿เแสMขJ
Cข@
:7
inputs+???????????????????????????`
p 
ช "2/+???????????????????????????`
5__inference_batch_normalization_10_layer_call_fn_6553W฿เแส2ข/
(ข%

inputs`
p
ช "`
5__inference_batch_normalization_10_layer_call_fn_6562W฿เแส2ข/
(ข%

inputs`
p 
ช "`ธ
P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_6589d๐๑๒ห2ข/
(ข%

inputs`
p
ช "$ข!

0`
 ธ
P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_6606d๐๑๒ห2ข/
(ข%

inputs`
p 
ช "$ข!

0`
 ๏
P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_6641๐๑๒หMขJ
Cข@
:7
inputs+???????????????????????????`
p
ช "?ข<
52
0+???????????????????????????`
 ๏
P__inference_batch_normalization_11_layer_call_and_return_conditional_losses_6658๐๑๒หMขJ
Cข@
:7
inputs+???????????????????????????`
p 
ช "?ข<
52
0+???????????????????????????`
 
5__inference_batch_normalization_11_layer_call_fn_6615W๐๑๒ห2ข/
(ข%

inputs`
p
ช "`
5__inference_batch_normalization_11_layer_call_fn_6624W๐๑๒ห2ข/
(ข%

inputs`
p 
ช "`ว
5__inference_batch_normalization_11_layer_call_fn_6667๐๑๒หMขJ
Cข@
:7
inputs+???????????????????????????`
p
ช "2/+???????????????????????????`ว
5__inference_batch_normalization_11_layer_call_fn_6676๐๑๒หMขJ
Cข@
:7
inputs+???????????????????????????`
p 
ช "2/+???????????????????????????`ธ
P__inference_batch_normalization_12_layer_call_and_return_conditional_losses_6717d2ข/
(ข%

inputs@
p
ช "$ข!

0@
 ธ
P__inference_batch_normalization_12_layer_call_and_return_conditional_losses_6735d2ข/
(ข%

inputs@
p 
ช "$ข!

0@
 ๏
P__inference_batch_normalization_12_layer_call_and_return_conditional_losses_6771MขJ
Cข@
:7
inputs+???????????????????????????@
p
ช "?ข<
52
0+???????????????????????????@
 ๏
P__inference_batch_normalization_12_layer_call_and_return_conditional_losses_6789MขJ
Cข@
:7
inputs+???????????????????????????@
p 
ช "?ข<
52
0+???????????????????????????@
 
5__inference_batch_normalization_12_layer_call_fn_6744W2ข/
(ข%

inputs@
p
ช "@
5__inference_batch_normalization_12_layer_call_fn_6753W2ข/
(ข%

inputs@
p 
ช "@ว
5__inference_batch_normalization_12_layer_call_fn_6798MขJ
Cข@
:7
inputs+???????????????????????????@
p
ช "2/+???????????????????????????@ว
5__inference_batch_normalization_12_layer_call_fn_6807MขJ
Cข@
:7
inputs+???????????????????????????@
p 
ช "2/+???????????????????????????@๋
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5377KLMฤMขJ
Cข@
:7
inputs+???????????????????????????0
p
ช "?ข<
52
0+???????????????????????????0
 ๋
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5394KLMฤMขJ
Cข@
:7
inputs+???????????????????????????0
p 
ช "?ข<
52
0+???????????????????????????0
 ด
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5429aKLMฤ2ข/
(ข%

inputs0
p
ช "$ข!

00
 ด
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5446aKLMฤ2ข/
(ข%

inputs0
p 
ช "$ข!

00
 ร
4__inference_batch_normalization_1_layer_call_fn_5403KLMฤMขJ
Cข@
:7
inputs+???????????????????????????0
p
ช "2/+???????????????????????????0ร
4__inference_batch_normalization_1_layer_call_fn_5412KLMฤMขJ
Cข@
:7
inputs+???????????????????????????0
p 
ช "2/+???????????????????????????0
4__inference_batch_normalization_1_layer_call_fn_5455TKLMฤ2ข/
(ข%

inputs0
p
ช "0
4__inference_batch_normalization_1_layer_call_fn_5464TKLMฤ2ข/
(ข%

inputs0
p 
ช "0๋
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5491\]^ลMขJ
Cข@
:7
inputs+???????????????????????????0
p
ช "?ข<
52
0+???????????????????????????0
 ๋
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5508\]^ลMขJ
Cข@
:7
inputs+???????????????????????????0
p 
ช "?ข<
52
0+???????????????????????????0
 ด
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5543a\]^ล2ข/
(ข%

inputs0
p
ช "$ข!

00
 ด
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5560a\]^ล2ข/
(ข%

inputs0
p 
ช "$ข!

00
 ร
4__inference_batch_normalization_2_layer_call_fn_5517\]^ลMขJ
Cข@
:7
inputs+???????????????????????????0
p
ช "2/+???????????????????????????0ร
4__inference_batch_normalization_2_layer_call_fn_5526\]^ลMขJ
Cข@
:7
inputs+???????????????????????????0
p 
ช "2/+???????????????????????????0
4__inference_batch_normalization_2_layer_call_fn_5569T\]^ล2ข/
(ข%

inputs0
p
ช "0
4__inference_batch_normalization_2_layer_call_fn_5578T\]^ล2ข/
(ข%

inputs0
p 
ช "0ณ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_5619`mnop2ข/
(ข%

inputs 
p
ช "$ข!

0 
 ณ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_5637`mnop2ข/
(ข%

inputs 
p 
ช "$ข!

0 
 ๊
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_5673mnopMขJ
Cข@
:7
inputs+??????????????????????????? 
p
ช "?ข<
52
0+??????????????????????????? 
 ๊
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_5691mnopMขJ
Cข@
:7
inputs+??????????????????????????? 
p 
ช "?ข<
52
0+??????????????????????????? 
 
4__inference_batch_normalization_3_layer_call_fn_5646Smnop2ข/
(ข%

inputs 
p
ช " 
4__inference_batch_normalization_3_layer_call_fn_5655Smnop2ข/
(ข%

inputs 
p 
ช " ย
4__inference_batch_normalization_3_layer_call_fn_5700mnopMขJ
Cข@
:7
inputs+??????????????????????????? 
p
ช "2/+??????????????????????????? ย
4__inference_batch_normalization_3_layer_call_fn_5709mnopMขJ
Cข@
:7
inputs+??????????????????????????? 
p 
ช "2/+??????????????????????????? ถ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5751cฦ2ข/
(ข%

inputs0
p
ช "$ข!

00
 ถ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5768cฦ2ข/
(ข%

inputs0
p 
ช "$ข!

00
 ํ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5803ฦMขJ
Cข@
:7
inputs+???????????????????????????0
p
ช "?ข<
52
0+???????????????????????????0
 ํ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5820ฦMขJ
Cข@
:7
inputs+???????????????????????????0
p 
ช "?ข<
52
0+???????????????????????????0
 
4__inference_batch_normalization_4_layer_call_fn_5777Vฦ2ข/
(ข%

inputs0
p
ช "0
4__inference_batch_normalization_4_layer_call_fn_5786Vฦ2ข/
(ข%

inputs0
p 
ช "0ล
4__inference_batch_normalization_4_layer_call_fn_5829ฦMขJ
Cข@
:7
inputs+???????????????????????????0
p
ช "2/+???????????????????????????0ล
4__inference_batch_normalization_4_layer_call_fn_5838ฦMขJ
Cข@
:7
inputs+???????????????????????????0
p 
ช "2/+???????????????????????????0ท
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_5865dว2ข/
(ข%

inputs0
p
ช "$ข!

00
 ท
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_5882dว2ข/
(ข%

inputs0
p 
ช "$ข!

00
 ๎
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_5917วMขJ
Cข@
:7
inputs+???????????????????????????0
p
ช "?ข<
52
0+???????????????????????????0
 ๎
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_5934วMขJ
Cข@
:7
inputs+???????????????????????????0
p 
ช "?ข<
52
0+???????????????????????????0
 
4__inference_batch_normalization_5_layer_call_fn_5891Wว2ข/
(ข%

inputs0
p
ช "0
4__inference_batch_normalization_5_layer_call_fn_5900Wว2ข/
(ข%

inputs0
p 
ช "0ฦ
4__inference_batch_normalization_5_layer_call_fn_5943วMขJ
Cข@
:7
inputs+???????????????????????????0
p
ช "2/+???????????????????????????0ฦ
4__inference_batch_normalization_5_layer_call_fn_5952วMขJ
Cข@
:7
inputs+???????????????????????????0
p 
ช "2/+???????????????????????????0ท
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_5993dกขฃค2ข/
(ข%

inputs 
p
ช "$ข!

0 
 ท
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_6011dกขฃค2ข/
(ข%

inputs 
p 
ช "$ข!

0 
 ๎
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_6047กขฃคMขJ
Cข@
:7
inputs+??????????????????????????? 
p
ช "?ข<
52
0+??????????????????????????? 
 ๎
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_6065กขฃคMขJ
Cข@
:7
inputs+??????????????????????????? 
p 
ช "?ข<
52
0+??????????????????????????? 
 
4__inference_batch_normalization_6_layer_call_fn_6020Wกขฃค2ข/
(ข%

inputs 
p
ช " 
4__inference_batch_normalization_6_layer_call_fn_6029Wกขฃค2ข/
(ข%

inputs 
p 
ช " ฦ
4__inference_batch_normalization_6_layer_call_fn_6074กขฃคMขJ
Cข@
:7
inputs+??????????????????????????? 
p
ช "2/+??????????????????????????? ฦ
4__inference_batch_normalization_6_layer_call_fn_6083กขฃคMขJ
Cข@
:7
inputs+??????????????????????????? 
p 
ช "2/+??????????????????????????? ท
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_6113dฏฐฑศ2ข/
(ข%

inputs0
p
ช "$ข!

00
 ท
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_6130dฏฐฑศ2ข/
(ข%

inputs0
p 
ช "$ข!

00
 ๎
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_6165ฏฐฑศMขJ
Cข@
:7
inputs+???????????????????????????0
p
ช "?ข<
52
0+???????????????????????????0
 ๎
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_6182ฏฐฑศMขJ
Cข@
:7
inputs+???????????????????????????0
p 
ช "?ข<
52
0+???????????????????????????0
 
4__inference_batch_normalization_7_layer_call_fn_6139Wฏฐฑศ2ข/
(ข%

inputs0
p
ช "0
4__inference_batch_normalization_7_layer_call_fn_6148Wฏฐฑศ2ข/
(ข%

inputs0
p 
ช "0ฦ
4__inference_batch_normalization_7_layer_call_fn_6191ฏฐฑศMขJ
Cข@
:7
inputs+???????????????????????????0
p
ช "2/+???????????????????????????0ฦ
4__inference_batch_normalization_7_layer_call_fn_6200ฏฐฑศMขJ
Cข@
:7
inputs+???????????????????????????0
p 
ช "2/+???????????????????????????0ท
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_6227dภมยษ2ข/
(ข%

inputs0
p
ช "$ข!

00
 ท
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_6244dภมยษ2ข/
(ข%

inputs0
p 
ช "$ข!

00
 ๎
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_6279ภมยษMขJ
Cข@
:7
inputs+???????????????????????????0
p
ช "?ข<
52
0+???????????????????????????0
 ๎
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_6296ภมยษMขJ
Cข@
:7
inputs+???????????????????????????0
p 
ช "?ข<
52
0+???????????????????????????0
 
4__inference_batch_normalization_8_layer_call_fn_6253Wภมยษ2ข/
(ข%

inputs0
p
ช "0
4__inference_batch_normalization_8_layer_call_fn_6262Wภมยษ2ข/
(ข%

inputs0
p 
ช "0ฦ
4__inference_batch_normalization_8_layer_call_fn_6305ภมยษMขJ
Cข@
:7
inputs+???????????????????????????0
p
ช "2/+???????????????????????????0ฦ
4__inference_batch_normalization_8_layer_call_fn_6314ภมยษMขJ
Cข@
:7
inputs+???????????????????????????0
p 
ช "2/+???????????????????????????0๎
O__inference_batch_normalization_9_layer_call_and_return_conditional_losses_6355ัาำิMขJ
Cข@
:7
inputs+???????????????????????????@
p
ช "?ข<
52
0+???????????????????????????@
 ๎
O__inference_batch_normalization_9_layer_call_and_return_conditional_losses_6373ัาำิMขJ
Cข@
:7
inputs+???????????????????????????@
p 
ช "?ข<
52
0+???????????????????????????@
 ท
O__inference_batch_normalization_9_layer_call_and_return_conditional_losses_6409dัาำิ2ข/
(ข%

inputs@
p
ช "$ข!

0@
 ท
O__inference_batch_normalization_9_layer_call_and_return_conditional_losses_6427dัาำิ2ข/
(ข%

inputs@
p 
ช "$ข!

0@
 ฦ
4__inference_batch_normalization_9_layer_call_fn_6382ัาำิMขJ
Cข@
:7
inputs+???????????????????????????@
p
ช "2/+???????????????????????????@ฦ
4__inference_batch_normalization_9_layer_call_fn_6391ัาำิMขJ
Cข@
:7
inputs+???????????????????????????@
p 
ช "2/+???????????????????????????@
4__inference_batch_normalization_9_layer_call_fn_6436Wัาำิ2ข/
(ข%

inputs@
p
ช "@
4__inference_batch_normalization_9_layer_call_fn_6445Wัาำิ2ข/
(ข%

inputs@
p 
ช "@้
M__inference_batch_normalization_layer_call_and_return_conditional_losses_5250:;<รMขJ
Cข@
:7
inputs+??????????????????????????? 
p
ช "?ข<
52
0+??????????????????????????? 
 ้
M__inference_batch_normalization_layer_call_and_return_conditional_losses_5267:;<รMขJ
Cข@
:7
inputs+??????????????????????????? 
p 
ช "?ข<
52
0+??????????????????????????? 
 ฒ
M__inference_batch_normalization_layer_call_and_return_conditional_losses_5302a:;<ร2ข/
(ข%

inputs 
p
ช "$ข!

0 
 ฒ
M__inference_batch_normalization_layer_call_and_return_conditional_losses_5319a:;<ร2ข/
(ข%

inputs 
p 
ช "$ข!

0 
 ม
2__inference_batch_normalization_layer_call_fn_5276:;<รMขJ
Cข@
:7
inputs+??????????????????????????? 
p
ช "2/+??????????????????????????? ม
2__inference_batch_normalization_layer_call_fn_5285:;<รMขJ
Cข@
:7
inputs+??????????????????????????? 
p 
ช "2/+??????????????????????????? 
2__inference_batch_normalization_layer_call_fn_5328T:;<ร2ข/
(ข%

inputs 
p
ช " 
2__inference_batch_normalization_layer_call_fn_5337T:;<ร2ข/
(ข%

inputs 
p 
ช " 
B__inference_conv2d_1_layer_call_and_return_conditional_losses_5354YE.ข+
$ข!

inputs 
ช "$ข!

00
 w
'__inference_conv2d_1_layer_call_fn_5360LE.ข+
$ข!

inputs 
ช "0
B__inference_conv2d_2_layer_call_and_return_conditional_losses_5595Yg.ข+
$ข!

inputs0
ช "$ข!

0 
 w
'__inference_conv2d_2_layer_call_fn_5601Lg.ข+
$ข!

inputs0
ช " 
B__inference_conv2d_3_layer_call_and_return_conditional_losses_5728Yy.ข+
$ข!

inputs 
ช "$ข!

00
 w
'__inference_conv2d_3_layer_call_fn_5734Ly.ข+
$ข!

inputs 
ช "0?
B__inference_conv2d_4_layer_call_and_return_conditional_losses_5969Z.ข+
$ข!

inputs0
ช "$ข!

0 
 x
'__inference_conv2d_4_layer_call_fn_5975M.ข+
$ข!

inputs0
ช " ?
B__inference_conv2d_5_layer_call_and_return_conditional_losses_6090Zฉ.ข+
$ข!

inputs 
ช "$ข!

00
 x
'__inference_conv2d_5_layer_call_fn_6096Mฉ.ข+
$ข!

inputs 
ช "0?
B__inference_conv2d_6_layer_call_and_return_conditional_losses_6331Zห.ข+
$ข!

inputs0
ช "$ข!

0@
 x
'__inference_conv2d_6_layer_call_fn_6337Mห.ข+
$ข!

inputs0
ช "@?
B__inference_conv2d_7_layer_call_and_return_conditional_losses_6452Zู.ข+
$ข!

inputs@
ช "$ข!

0`
 x
'__inference_conv2d_7_layer_call_fn_6458Mู.ข+
$ข!

inputs@
ช "`?
B__inference_conv2d_8_layer_call_and_return_conditional_losses_6693Z๛.ข+
$ข!

inputs`
ช "$ข!

0@
 x
'__inference_conv2d_8_layer_call_fn_6699M๛.ข+
$ข!

inputs`
ช "@
@__inference_conv2d_layer_call_and_return_conditional_losses_5227Y4.ข+
$ข!

inputs1(
ช "$ข!

0 
 u
%__inference_conv2d_layer_call_fn_5233L4.ข+
$ข!

inputs1(
ช " 
?__inference_dense_layer_call_and_return_conditional_losses_6856L&ข#
ข

inputs@
ช "ข

0
 g
$__inference_dense_layer_call_fn_6863?&ข#
ข

inputs@
ช "แ
L__inference_depthwise_conv2d_1_layer_call_and_return_conditional_losses_1406IขF
?ข<
:7
inputs+???????????????????????????0
ช "?ข<
52
0+???????????????????????????0
 น
1__inference_depthwise_conv2d_1_layer_call_fn_1422IขF
?ข<
:7
inputs+???????????????????????????0
ช "2/+???????????????????????????0แ
L__inference_depthwise_conv2d_2_layer_call_and_return_conditional_losses_1700บIขF
?ข<
:7
inputs+???????????????????????????0
ช "?ข<
52
0+???????????????????????????0
 น
1__inference_depthwise_conv2d_2_layer_call_fn_1716บIขF
?ข<
:7
inputs+???????????????????????????0
ช "2/+???????????????????????????0แ
L__inference_depthwise_conv2d_3_layer_call_and_return_conditional_losses_1994๊IขF
?ข<
:7
inputs+???????????????????????????`
ช "?ข<
52
0+???????????????????????????`
 น
1__inference_depthwise_conv2d_3_layer_call_fn_2010๊IขF
?ข<
:7
inputs+???????????????????????????`
ช "2/+???????????????????????????`?
J__inference_depthwise_conv2d_layer_call_and_return_conditional_losses_1112VIขF
?ข<
:7
inputs+???????????????????????????0
ช "?ข<
52
0+???????????????????????????0
 ถ
/__inference_depthwise_conv2d_layer_call_fn_1128VIขF
?ข<
:7
inputs+???????????????????????????0
ช "2/+???????????????????????????0
A__inference_dropout_layer_call_and_return_conditional_losses_6831J*ข'
 ข

inputs@
p
ช "ข

0@
 
A__inference_dropout_layer_call_and_return_conditional_losses_6836J*ข'
 ข

inputs@
p 
ช "ข

0@
 g
&__inference_dropout_layer_call_fn_6841=*ข'
 ข

inputs@
p
ช "@g
&__inference_dropout_layer_call_fn_6846=*ข'
 ข

inputs@
p 
ช "@
F__inference_functional_1_layer_call_and_return_conditional_losses_4176ัs4:;<รEKLMฤV\]^ลgmnopyฦวกขฃคฉฏฐฑศบภมยษหัาำิู฿เแส๊๐๑๒ห๛<ข9
2ข/
%"
input_1?????????1(
p

 
ช "ข

0
 
F__inference_functional_1_layer_call_and_return_conditional_losses_4422ัs4:;<รEKLMฤV\]^ลgmnopyฦวกขฃคฉฏฐฑศบภมยษหัาำิู฿เแส๊๐๑๒ห๛<ข9
2ข/
%"
input_1?????????1(
p 

 
ช "ข

0
 
F__inference_functional_1_layer_call_and_return_conditional_losses_4819ะs4:;<รEKLMฤV\]^ลgmnopyฦวกขฃคฉฏฐฑศบภมยษหัาำิู฿เแส๊๐๑๒ห๛;ข8
1ข.
$!
inputs?????????1(
p

 
ช "ข

0
 
F__inference_functional_1_layer_call_and_return_conditional_losses_5065ะs4:;<รEKLMฤV\]^ลgmnopyฦวกขฃคฉฏฐฑศบภมยษหัาำิู฿เแส๊๐๑๒ห๛;ข8
1ข.
$!
inputs?????????1(
p 

 
ช "ข

0
 ๔
+__inference_functional_1_layer_call_fn_4494ฤs4:;<รEKLMฤV\]^ลgmnopyฦวกขฃคฉฏฐฑศบภมยษหัาำิู฿เแส๊๐๑๒ห๛<ข9
2ข/
%"
input_1?????????1(
p

 
ช "๔
+__inference_functional_1_layer_call_fn_4566ฤs4:;<รEKLMฤV\]^ลgmnopyฦวกขฃคฉฏฐฑศบภมยษหัาำิู฿เแส๊๐๑๒ห๛<ข9
2ข/
%"
input_1?????????1(
p 

 
ช "๓
+__inference_functional_1_layer_call_fn_5137รs4:;<รEKLMฤV\]^ลgmnopyฦวกขฃคฉฏฐฑศบภมยษหัาำิู฿เแส๊๐๑๒ห๛;ข8
1ข.
$!
inputs?????????1(
p

 
ช "๓
+__inference_functional_1_layer_call_fn_5209รs4:;<รEKLMฤV\]^ลgmnopyฦวกขฃคฉฏฐฑศบภมยษหัาำิู฿เแส๊๐๑๒ห๛;ข8
1ข.
$!
inputs?????????1(
p 

 
ช "?
R__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_2197RขO
HขE
C@
inputs4????????????????????????????????????
ช ".ข+
$!
0??????????????????
 ฒ
7__inference_global_average_pooling2d_layer_call_fn_2209wRขO
HขE
C@
inputs4????????????????????????????????????
ช "!??????????????????
A__inference_re_lu_1_layer_call_and_return_conditional_losses_5469V.ข+
$ข!

inputs0
ช "$ข!

00
 s
&__inference_re_lu_1_layer_call_fn_5474I.ข+
$ข!

inputs0
ช "0
A__inference_re_lu_2_layer_call_and_return_conditional_losses_5583V.ข+
$ข!

inputs0
ช "$ข!

00
 s
&__inference_re_lu_2_layer_call_fn_5588I.ข+
$ข!

inputs0
ช "0
A__inference_re_lu_3_layer_call_and_return_conditional_losses_5843V.ข+
$ข!

inputs0
ช "$ข!

00
 s
&__inference_re_lu_3_layer_call_fn_5848I.ข+
$ข!

inputs0
ช "0
A__inference_re_lu_4_layer_call_and_return_conditional_losses_5957V.ข+
$ข!

inputs0
ช "$ข!

00
 s
&__inference_re_lu_4_layer_call_fn_5962I.ข+
$ข!

inputs0
ช "0
A__inference_re_lu_5_layer_call_and_return_conditional_losses_6205V.ข+
$ข!

inputs0
ช "$ข!

00
 s
&__inference_re_lu_5_layer_call_fn_6210I.ข+
$ข!

inputs0
ช "0
A__inference_re_lu_6_layer_call_and_return_conditional_losses_6319V.ข+
$ข!

inputs0
ช "$ข!

00
 s
&__inference_re_lu_6_layer_call_fn_6324I.ข+
$ข!

inputs0
ช "0
A__inference_re_lu_7_layer_call_and_return_conditional_losses_6567V.ข+
$ข!

inputs`
ช "$ข!

0`
 s
&__inference_re_lu_7_layer_call_fn_6572I.ข+
$ข!

inputs`
ช "`
A__inference_re_lu_8_layer_call_and_return_conditional_losses_6681V.ข+
$ข!

inputs`
ช "$ข!

0`
 s
&__inference_re_lu_8_layer_call_fn_6686I.ข+
$ข!

inputs`
ช "`
?__inference_re_lu_layer_call_and_return_conditional_losses_5342V.ข+
$ข!

inputs 
ช "$ข!

0 
 q
$__inference_re_lu_layer_call_fn_5347I.ข+
$ข!

inputs 
ช " ๚
"__inference_signature_wrapper_3923ำs4:;<รEKLMฤV\]^ลgmnopyฦวกขฃคฉฏฐฑศบภมยษหัาำิู฿เแส๊๐๑๒ห๛6ข3
ข 
,ช)
'
input_1
input_11("$ช!

dense
denseฆ
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_5215R*ข'
 ข

inputs1(
ช "$ข!

01(
 ~
5__inference_tf_op_layer_ExpandDims_layer_call_fn_5220E*ข'
 ข

inputs1(
ช "1(