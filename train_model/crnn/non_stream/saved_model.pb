Ί
Ρ£
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
Ύ
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
 "serve*2.3.02v2.3.0-rc2-23-gb36436b0878υπ
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:*
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:*
dtype0
y
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namedense_2/kernel
r
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes
:	*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0

stream/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_namestream/conv2d/kernel

(stream/conv2d/kernel/Read/ReadVariableOpReadVariableOpstream/conv2d/kernel*&
_output_shapes
:*
dtype0
|
stream/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_namestream/conv2d/bias
u
&stream/conv2d/bias/Read/ReadVariableOpReadVariableOpstream/conv2d/bias*
_output_shapes
:*
dtype0

stream_1/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namestream_1/conv2d_1/kernel

,stream_1/conv2d_1/kernel/Read/ReadVariableOpReadVariableOpstream_1/conv2d_1/kernel*&
_output_shapes
:*
dtype0

stream_1/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namestream_1/conv2d_1/bias
}
*stream_1/conv2d_1/bias/Read/ReadVariableOpReadVariableOpstream_1/conv2d_1/bias*
_output_shapes
:*
dtype0

gru/cell/gru_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ΐ*)
shared_namegru/cell/gru_cell/kernel

,gru/cell/gru_cell/kernel/Read/ReadVariableOpReadVariableOpgru/cell/gru_cell/kernel* 
_output_shapes
:
ΐ*
dtype0
’
"gru/cell/gru_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*3
shared_name$"gru/cell/gru_cell/recurrent_kernel

6gru/cell/gru_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp"gru/cell/gru_cell/recurrent_kernel* 
_output_shapes
:
*
dtype0

gru/cell/gru_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_namegru/cell/gru_cell/bias

*gru/cell/gru_cell/bias/Read/ReadVariableOpReadVariableOpgru/cell/gru_cell/bias*
_output_shapes
:	*
dtype0

gru/cell/VariableVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*"
shared_namegru/cell/Variable
x
%gru/cell/Variable/Read/ReadVariableOpReadVariableOpgru/cell/Variable*
_output_shapes
:	*
dtype0

NoOpNoOp
7
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ε6
value»6BΈ6 B±6

layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
	variables
regularization_losses
trainable_variables
	keras_api

signatures
 
R
	variables
regularization_losses
trainable_variables
	keras_api
m
cell
state_shape
	variables
regularization_losses
trainable_variables
	keras_api
m
cell
state_shape
	variables
regularization_losses
trainable_variables
 	keras_api
R
!	variables
"regularization_losses
#trainable_variables
$	keras_api
[
%gru
&	variables
'regularization_losses
(trainable_variables
)	keras_api
m
*cell
+state_shape
,	variables
-regularization_losses
.trainable_variables
/	keras_api
R
0	variables
1regularization_losses
2trainable_variables
3	keras_api
h

4kernel
5bias
6	variables
7regularization_losses
8trainable_variables
9	keras_api
h

:kernel
;bias
<	variables
=regularization_losses
>trainable_variables
?	keras_api
h

@kernel
Abias
B	variables
Cregularization_losses
Dtrainable_variables
E	keras_api
^
F0
G1
H2
I3
J4
K5
L6
47
58
:9
;10
@11
A12
 
^
F0
G1
H2
I3
J4
K5
L6
47
58
:9
;10
@11
A12
­
Mlayer_metrics

Nlayers
	variables
Ometrics
Pnon_trainable_variables
regularization_losses
Qlayer_regularization_losses
trainable_variables
 
 
 
 
­
Rlayer_metrics

Slayers
	variables
Tmetrics
Unon_trainable_variables
regularization_losses
Vlayer_regularization_losses
trainable_variables
h

Fkernel
Gbias
W	variables
Xregularization_losses
Ytrainable_variables
Z	keras_api
 

F0
G1
 

F0
G1
­
[layer_metrics

\layers
	variables
]metrics
^non_trainable_variables
regularization_losses
_layer_regularization_losses
trainable_variables
h

Hkernel
Ibias
`	variables
aregularization_losses
btrainable_variables
c	keras_api
 

H0
I1
 

H0
I1
­
dlayer_metrics

elayers
	variables
fmetrics
gnon_trainable_variables
regularization_losses
hlayer_regularization_losses
trainable_variables
 
 
 
­
ilayer_metrics

jlayers
!	variables
kmetrics
lnon_trainable_variables
"regularization_losses
mlayer_regularization_losses
#trainable_variables
l
ncell
o
state_spec
p	variables
qregularization_losses
rtrainable_variables
s	keras_api

J0
K1
L2
 

J0
K1
L2
­
tlayer_metrics

ulayers
&	variables
vmetrics
wnon_trainable_variables
'regularization_losses
xlayer_regularization_losses
(trainable_variables
R
y	variables
zregularization_losses
{trainable_variables
|	keras_api
 
 
 
 
―
}layer_metrics

~layers
,	variables
metrics
non_trainable_variables
-regularization_losses
 layer_regularization_losses
.trainable_variables
 
 
 
²
layer_metrics
layers
0	variables
metrics
non_trainable_variables
1regularization_losses
 layer_regularization_losses
2trainable_variables
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

40
51
 

40
51
²
layer_metrics
layers
6	variables
metrics
non_trainable_variables
7regularization_losses
 layer_regularization_losses
8trainable_variables
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

:0
;1
 

:0
;1
²
layer_metrics
layers
<	variables
metrics
non_trainable_variables
=regularization_losses
 layer_regularization_losses
>trainable_variables
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

@0
A1
 

@0
A1
²
layer_metrics
layers
B	variables
metrics
non_trainable_variables
Cregularization_losses
 layer_regularization_losses
Dtrainable_variables
PN
VARIABLE_VALUEstream/conv2d/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEstream/conv2d/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEstream_1/conv2d_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEstream_1/conv2d_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEgru/cell/gru_cell/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE"gru/cell/gru_cell/recurrent_kernel&variables/5/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEgru/cell/gru_cell/bias&variables/6/.ATTRIBUTES/VARIABLE_VALUE
 
N
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
 
 
 
 
 
 
 
 

F0
G1
 

F0
G1
²
layer_metrics
layers
W	variables
metrics
non_trainable_variables
Xregularization_losses
 layer_regularization_losses
Ytrainable_variables
 

0
 
 
 

H0
I1
 

H0
I1
²
layer_metrics
layers
`	variables
metrics
non_trainable_variables
aregularization_losses
 layer_regularization_losses
btrainable_variables
 

0
 
 
 
 
 
 
 
 


Jkernel
Krecurrent_kernel
Lbias
 	variables
‘regularization_losses
’trainable_variables
£	keras_api
 

J0
K1
L2
 

J0
K1
L2
Ώ
€layer_metrics
₯layers
p	variables
¦states
§metrics
¨non_trainable_variables
qregularization_losses
 ©layer_regularization_losses
rtrainable_variables
 

%0
 
 
 
 
 
 
²
ͺlayer_metrics
«layers
y	variables
¬metrics
­non_trainable_variables
zregularization_losses
 ?layer_regularization_losses
{trainable_variables
 

*0
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
 
 
 
 
 
 
 
 
 

J0
K1
L2
 

J0
K1
L2
΅
―layer_metrics
°layers
 	variables
±metrics
²non_trainable_variables
‘regularization_losses
 ³layer_regularization_losses
’trainable_variables
 

n0

΄0
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
mk
VARIABLE_VALUEgru/cell/VariableFlayer_with_weights-2/gru/keras_api/states/0/.ATTRIBUTES/VARIABLE_VALUE
p
serving_default_input_1Placeholder*"
_output_shapes
:1(*
dtype0*
shape:1(
έ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1stream/conv2d/kernelstream/conv2d/biasstream_1/conv2d_1/kernelstream_1/conv2d_1/biasgru/cell/gru_cell/biasgru/cell/gru_cell/kernelgru/cell/Variable"gru/cell/gru_cell/recurrent_kerneldense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*/
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference_signature_wrapper_2138
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ϊ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp(stream/conv2d/kernel/Read/ReadVariableOp&stream/conv2d/bias/Read/ReadVariableOp,stream_1/conv2d_1/kernel/Read/ReadVariableOp*stream_1/conv2d_1/bias/Read/ReadVariableOp,gru/cell/gru_cell/kernel/Read/ReadVariableOp6gru/cell/gru_cell/recurrent_kernel/Read/ReadVariableOp*gru/cell/gru_cell/bias/Read/ReadVariableOp%gru/cell/Variable/Read/ReadVariableOpConst*
Tin
2*
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
__inference__traced_save_4180
½
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasstream/conv2d/kernelstream/conv2d/biasstream_1/conv2d_1/kernelstream_1/conv2d_1/biasgru/cell/gru_cell/kernel"gru/cell/gru_cell/recurrent_kernelgru/cell/gru_cell/biasgru/cell/Variable*
Tin
2*
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
 __inference__traced_restore_4232
€
_
A__inference_dropout_layer_call_and_return_conditional_losses_3439

inputs

identity_1R
IdentityIdentityinputs*
T0*
_output_shapes
:	2

Identitya

Identity_1IdentityIdentity:output:0*
T0*
_output_shapes
:	2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs
ό
Τ
"__inference_gru_layer_call_fn_3402

inputs
gru_cell_gru_cell_bias
gru_cell_gru_cell_kernel
gru_cell_variable&
"gru_cell_gru_cell_recurrent_kernel
identity’StatefulPartitionedCallΑ
StatefulPartitionedCallStatefulPartitionedCallinputsgru_cell_gru_cell_biasgru_cell_gru_cell_kernelgru_cell_variable"gru_cell_gru_cell_recurrent_kernel*
Tin	
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *F
fAR?
=__inference_gru_layer_call_and_return_conditional_losses_18572
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:2

Identity"
identityIdentity:output:0*2
_input_shapes!
:+ΐ::::22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:+ΐ
 
_user_specified_nameinputs


A__inference_gru_cell_layer_call_and_return_conditional_losses_981

inputs

states)
%readvariableop_gru_cell_gru_cell_bias2
.matmul_readvariableop_gru_cell_gru_cell_kernel@
<matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel
identity

identity_1
ReadVariableOpReadVariableOp%readvariableop_gru_cell_gru_cell_bias*
_output_shapes
:	*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOp.matmul_readvariableop_gru_cell_gru_cell_kernel* 
_output_shapes
:
ΐ*
dtype02
MatMul/ReadVariableOpk
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
MatMulk
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*
_output_shapes
:	2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split/split_dim
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
splitz
MatMul_1/ReadVariableOpReadVariableOpstates*
_output_shapes
:	*
dtype02
MatMul_1/ReadVariableOp΅
MatMul_1/ReadVariableOp_1ReadVariableOp<matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOp_1
MatMul_1MatMulMatMul_1/ReadVariableOp:value:0!MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
:	2

MatMul_1q
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*
_output_shapes
:	2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ????2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split_1/split_dim»
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2	
split_1_
addAddV2split:output:0split_1:output:0*
T0*
_output_shapes
:	2
addP
SigmoidSigmoidadd:z:0*
T0*
_output_shapes
:	2	
Sigmoidc
add_1AddV2split:output:1split_1:output:1*
T0*
_output_shapes
:	2
add_1V
	Sigmoid_1Sigmoid	add_1:z:0*
T0*
_output_shapes
:	2
	Sigmoid_1\
mulMulSigmoid_1:y:0split_1:output:2*
T0*
_output_shapes
:	2
mulZ
add_2AddV2split:output:2mul:z:0*
T0*
_output_shapes
:	2
add_2I
TanhTanh	add_2:z:0*
T0*
_output_shapes
:	2
Tanht
mul_1/ReadVariableOpReadVariableOpstates*
_output_shapes
:	*
dtype02
mul_1/ReadVariableOpj
mul_1MulSigmoid:y:0mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xX
subSubsub/x:output:0Sigmoid:y:0*
T0*
_output_shapes
:	2
subR
mul_2Mulsub:z:0Tanh:y:0*
T0*
_output_shapes
:	2
mul_2W
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*
_output_shapes
:	2
add_3U
IdentityIdentity	add_3:z:0*
T0*
_output_shapes
:	2

IdentityY

Identity_1Identity	add_3:z:0*
T0*
_output_shapes
:	2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*.
_input_shapes
:	ΐ:::::G C

_output_shapes
:	ΐ
 
_user_specified_nameinputs:&"
 
_user_specified_namestates
Ύ	
ΰ
'__inference_gru_cell_layer_call_fn_4013

inputs
states_0
gru_cell_gru_cell_bias
gru_cell_gru_cell_kernel&
"gru_cell_gru_cell_recurrent_kernel
identity

identity_1’StatefulPartitionedCallΈ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0gru_cell_gru_cell_biasgru_cell_gru_cell_kernel"gru_cell_gru_cell_recurrent_kernel*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

::*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_gru_cell_layer_call_and_return_conditional_losses_40052
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*.
_input_shapes
:	ΐ::::22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	ΐ
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0
Ύ

$__inference_dense_layer_call_fn_3466

inputs
dense_kernel

dense_bias
identity’StatefulPartitionedCallν
StatefulPartitionedCallStatefulPartitionedCallinputsdense_kernel
dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_19362
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*&
_input_shapes
:	::22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	
 
_user_specified_nameinputs


Μ
@__inference_stream_layer_call_and_return_conditional_losses_1497

inputs5
1conv2d_conv2d_readvariableop_stream_conv2d_kernel4
0conv2d_biasadd_readvariableop_stream_conv2d_bias
identityΆ
conv2d/Conv2D/ReadVariableOpReadVariableOp1conv2d_conv2d_readvariableop_stream_conv2d_kernel*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp°
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:/&*
paddingVALID*
strides
2
conv2d/Conv2D«
conv2d/BiasAdd/ReadVariableOpReadVariableOp0conv2d_biasadd_readvariableop_stream_conv2d_bias*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:/&2
conv2d/BiasAddl
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*&
_output_shapes
:/&2
conv2d/Relul
IdentityIdentityconv2d/Relu:activations:0*
T0*&
_output_shapes
:/&2

Identity"
identityIdentity:output:0*-
_input_shapes
:1(:::N J
&
_output_shapes
:1(
 
_user_specified_nameinputs
Ή

Ϊ
B__inference_stream_1_layer_call_and_return_conditional_losses_3064

inputs;
7conv2d_1_conv2d_readvariableop_stream_1_conv2d_1_kernel:
6conv2d_1_biasadd_readvariableop_stream_1_conv2d_1_bias
identityΐ
conv2d_1/Conv2D/ReadVariableOpReadVariableOp7conv2d_1_conv2d_readvariableop_stream_1_conv2d_1_kernel*&
_output_shapes
:*
dtype02 
conv2d_1/Conv2D/ReadVariableOpΆ
conv2d_1/Conv2DConv2Dinputs&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:+$*
paddingVALID*
strides
2
conv2d_1/Conv2D΅
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp6conv2d_1_biasadd_readvariableop_stream_1_conv2d_1_bias*
_output_shapes
:*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp£
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:+$2
conv2d_1/BiasAddr
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*&
_output_shapes
:+$2
conv2d_1/Relun
IdentityIdentityconv2d_1/Relu:activations:0*
T0*&
_output_shapes
:+$2

Identity"
identityIdentity:output:0*-
_input_shapes
:/&:::N J
&
_output_shapes
:/&
 
_user_specified_nameinputs
ϋZ

=__inference_gru_layer_call_and_return_conditional_losses_3393

inputs7
3cell_gru_cell_readvariableop_gru_cell_gru_cell_bias@
<cell_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel;
7cell_gru_cell_matmul_1_readvariableop_gru_cell_variableN
Jcell_gru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel
identity’cell/AssignVariableOp’
cell/while
cell/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
cell/transpose/perm
cell/transpose	Transposeinputscell/transpose/perm:output:0*
T0*#
_output_shapes
:+ΐ2
cell/transposem

cell/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"+      @  2

cell/Shape~
cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
cell/strided_slice/stack
cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cell/strided_slice/stack_1
cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cell/strided_slice/stack_2
cell/strided_sliceStridedSlicecell/Shape:output:0!cell/strided_slice/stack:output:0#cell/strided_slice/stack_1:output:0#cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cell/strided_slice
 cell/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 cell/TensorArrayV2/element_shapeΔ
cell/TensorArrayV2TensorListReserve)cell/TensorArrayV2/element_shape:output:0cell/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
cell/TensorArrayV2Ι
:cell/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   @  2<
:cell/TensorArrayUnstack/TensorListFromTensor/element_shape
,cell/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorcell/transpose:y:0Ccell/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02.
,cell/TensorArrayUnstack/TensorListFromTensor
cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
cell/strided_slice_1/stack
cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cell/strided_slice_1/stack_1
cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cell/strided_slice_1/stack_2
cell/strided_slice_1StridedSlicecell/transpose:y:0#cell/strided_slice_1/stack:output:0%cell/strided_slice_1/stack_1:output:0%cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ΐ*
shrink_axis_mask2
cell/strided_slice_1±
cell/gru_cell/ReadVariableOpReadVariableOp3cell_gru_cell_readvariableop_gru_cell_gru_cell_bias*
_output_shapes
:	*
dtype02
cell/gru_cell/ReadVariableOp
cell/gru_cell/unstackUnpack$cell/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
cell/gru_cell/unstackΙ
#cell/gru_cell/MatMul/ReadVariableOpReadVariableOp<cell_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel* 
_output_shapes
:
ΐ*
dtype02%
#cell/gru_cell/MatMul/ReadVariableOp¬
cell/gru_cell/MatMulMatMulcell/strided_slice_1:output:0+cell/gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
cell/gru_cell/MatMul£
cell/gru_cell/BiasAddBiasAddcell/gru_cell/MatMul:product:0cell/gru_cell/unstack:output:0*
T0*
_output_shapes
:	2
cell/gru_cell/BiasAddl
cell/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
cell/gru_cell/Const
cell/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
cell/gru_cell/split/split_dimΜ
cell/gru_cell/splitSplit&cell/gru_cell/split/split_dim:output:0cell/gru_cell/BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
cell/gru_cell/splitΗ
%cell/gru_cell/MatMul_1/ReadVariableOpReadVariableOp7cell_gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype02'
%cell/gru_cell/MatMul_1/ReadVariableOpί
'cell/gru_cell/MatMul_1/ReadVariableOp_1ReadVariableOpJcell_gru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel* 
_output_shapes
:
*
dtype02)
'cell/gru_cell/MatMul_1/ReadVariableOp_1Δ
cell/gru_cell/MatMul_1MatMul-cell/gru_cell/MatMul_1/ReadVariableOp:value:0/cell/gru_cell/MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
:	2
cell/gru_cell/MatMul_1©
cell/gru_cell/BiasAdd_1BiasAdd cell/gru_cell/MatMul_1:product:0cell/gru_cell/unstack:output:1*
T0*
_output_shapes
:	2
cell/gru_cell/BiasAdd_1
cell/gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ????2
cell/gru_cell/Const_1
cell/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
cell/gru_cell/split_1/split_dim
cell/gru_cell/split_1SplitV cell/gru_cell/BiasAdd_1:output:0cell/gru_cell/Const_1:output:0(cell/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2
cell/gru_cell/split_1
cell/gru_cell/addAddV2cell/gru_cell/split:output:0cell/gru_cell/split_1:output:0*
T0*
_output_shapes
:	2
cell/gru_cell/addz
cell/gru_cell/SigmoidSigmoidcell/gru_cell/add:z:0*
T0*
_output_shapes
:	2
cell/gru_cell/Sigmoid
cell/gru_cell/add_1AddV2cell/gru_cell/split:output:1cell/gru_cell/split_1:output:1*
T0*
_output_shapes
:	2
cell/gru_cell/add_1
cell/gru_cell/Sigmoid_1Sigmoidcell/gru_cell/add_1:z:0*
T0*
_output_shapes
:	2
cell/gru_cell/Sigmoid_1
cell/gru_cell/mulMulcell/gru_cell/Sigmoid_1:y:0cell/gru_cell/split_1:output:2*
T0*
_output_shapes
:	2
cell/gru_cell/mul
cell/gru_cell/add_2AddV2cell/gru_cell/split:output:2cell/gru_cell/mul:z:0*
T0*
_output_shapes
:	2
cell/gru_cell/add_2s
cell/gru_cell/TanhTanhcell/gru_cell/add_2:z:0*
T0*
_output_shapes
:	2
cell/gru_cell/TanhΑ
"cell/gru_cell/mul_1/ReadVariableOpReadVariableOp7cell_gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype02$
"cell/gru_cell/mul_1/ReadVariableOp’
cell/gru_cell/mul_1Mulcell/gru_cell/Sigmoid:y:0*cell/gru_cell/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
cell/gru_cell/mul_1o
cell/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
cell/gru_cell/sub/x
cell/gru_cell/subSubcell/gru_cell/sub/x:output:0cell/gru_cell/Sigmoid:y:0*
T0*
_output_shapes
:	2
cell/gru_cell/sub
cell/gru_cell/mul_2Mulcell/gru_cell/sub:z:0cell/gru_cell/Tanh:y:0*
T0*
_output_shapes
:	2
cell/gru_cell/mul_2
cell/gru_cell/add_3AddV2cell/gru_cell/mul_1:z:0cell/gru_cell/mul_2:z:0*
T0*
_output_shapes
:	2
cell/gru_cell/add_3
"cell/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2$
"cell/TensorArrayV2_1/element_shapeΚ
cell/TensorArrayV2_1TensorListReserve+cell/TensorArrayV2_1/element_shape:output:0cell/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
cell/TensorArrayV2_1X
	cell/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
	cell/time£
cell/ReadVariableOpReadVariableOp7cell_gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype02
cell/ReadVariableOp
cell/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
cell/while/maximum_iterationst
cell/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
cell/while/loop_counter

cell/whileWhile cell/while/loop_counter:output:0&cell/while/maximum_iterations:output:0cell/time:output:0cell/TensorArrayV2_1:handle:0cell/ReadVariableOp:value:0cell/strided_slice:output:0<cell/TensorArrayUnstack/TensorListFromTensor:output_handle:03cell_gru_cell_readvariableop_gru_cell_gru_cell_bias<cell_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernelJcell_gru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*1
_output_shapes
: : : : :	: : : : : *%
_read_only_resource_inputs
	* 
bodyR
cell_while_body_3301* 
condR
cell_while_cond_3300*0
output_shapes
: : : : :	: : : : : *
parallel_iterations 2

cell/whileΏ
5cell/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      27
5cell/TensorArrayV2Stack/TensorListStack/element_shapeτ
'cell/TensorArrayV2Stack/TensorListStackTensorListStackcell/while:output:3>cell/TensorArrayV2Stack/TensorListStack/element_shape:output:0*#
_output_shapes
:+*
element_dtype02)
'cell/TensorArrayV2Stack/TensorListStack
cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
cell/strided_slice_2/stack
cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
cell/strided_slice_2/stack_1
cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cell/strided_slice_2/stack_2°
cell/strided_slice_2StridedSlice0cell/TensorArrayV2Stack/TensorListStack:tensor:0#cell/strided_slice_2/stack:output:0%cell/strided_slice_2/stack_1:output:0%cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	*
shrink_axis_mask2
cell/strided_slice_2
cell/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
cell/transpose_1/perm±
cell/transpose_1	Transpose0cell/TensorArrayV2Stack/TensorListStack:tensor:0cell/transpose_1/perm:output:0*
T0*#
_output_shapes
:+2
cell/transpose_1p
cell/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
cell/runtime
cell/AssignVariableOpAssignVariableOp7cell_gru_cell_matmul_1_readvariableop_gru_cell_variablecell/while:output:4^cell/ReadVariableOp&^cell/gru_cell/MatMul_1/ReadVariableOp#^cell/gru_cell/mul_1/ReadVariableOp*
_output_shapes
 *
dtype02
cell/AssignVariableOpb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimscell/strided_slice_2:output:0ExpandDims/dim:output:0*
T0*#
_output_shapes
:2

ExpandDims
IdentityIdentityExpandDims:output:0^cell/AssignVariableOp^cell/while*
T0*#
_output_shapes
:2

Identity"
identityIdentity:output:0*2
_input_shapes!
:+ΐ::::2.
cell/AssignVariableOpcell/AssignVariableOp2

cell/while
cell/while:K G
#
_output_shapes
:+ΐ
 
_user_specified_nameinputs
Δ
ύ
cell_while_cond_3300&
"cell_while_cell_while_loop_counter,
(cell_while_cell_while_maximum_iterations
cell_while_placeholder
cell_while_placeholder_1
cell_while_placeholder_2&
"cell_while_less_cell_strided_slice<
8cell_while_cell_while_cond_3300___redundant_placeholder0<
8cell_while_cell_while_cond_3300___redundant_placeholder1<
8cell_while_cell_while_cond_3300___redundant_placeholder2<
8cell_while_cell_while_cond_3300___redundant_placeholder3
cell_while_identity

cell/while/LessLesscell_while_placeholder"cell_while_less_cell_strided_slice*
T0*
_output_shapes
: 2
cell/while/Lessl
cell/while/IdentityIdentitycell/while/Less:z:0*
T0
*
_output_shapes
: 2
cell/while/Identity"3
cell_while_identitycell/while/Identity:output:0*8
_input_shapes'
%: : : : :	: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
:


Μ
@__inference_stream_layer_call_and_return_conditional_losses_3046

inputs5
1conv2d_conv2d_readvariableop_stream_conv2d_kernel4
0conv2d_biasadd_readvariableop_stream_conv2d_bias
identityΆ
conv2d/Conv2D/ReadVariableOpReadVariableOp1conv2d_conv2d_readvariableop_stream_conv2d_kernel*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp°
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:/&*
paddingVALID*
strides
2
conv2d/Conv2D«
conv2d/BiasAdd/ReadVariableOpReadVariableOp0conv2d_biasadd_readvariableop_stream_conv2d_bias*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:/&2
conv2d/BiasAddl
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*&
_output_shapes
:/&2
conv2d/Relul
IdentityIdentityconv2d/Relu:activations:0*
T0*&
_output_shapes
:/&2

Identity"
identityIdentity:output:0*-
_input_shapes
:1(:::N J
&
_output_shapes
:1(
 
_user_specified_nameinputs
#
Ό
while_body_1301
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0+
'while_gru_cell_gru_cell_gru_cell_bias_0-
)while_gru_cell_gru_cell_gru_cell_kernel_07
3while_gru_cell_gru_cell_gru_cell_recurrent_kernel_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor)
%while_gru_cell_gru_cell_gru_cell_bias+
'while_gru_cell_gru_cell_gru_cell_kernel5
1while_gru_cell_gru_cell_gru_cell_recurrent_kernel’&while/gru_cell/StatefulPartitionedCallΓ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   @  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeΛ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes
:	ΐ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemΛ
&while/gru_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2'while_gru_cell_gru_cell_gru_cell_bias_0)while_gru_cell_gru_cell_gru_cell_kernel_03while_gru_cell_gru_cell_gru_cell_recurrent_kernel_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:	:	*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_gru_cell_layer_call_and_return_conditional_losses_10772(
&while/gru_cell/StatefulPartitionedCallσ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder/while/gru_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:0'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2Ά
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3΄
while/Identity_4Identity/while/gru_cell/StatefulPartitionedCall:output:1'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
:	2
while/Identity_4"P
%while_gru_cell_gru_cell_gru_cell_bias'while_gru_cell_gru_cell_gru_cell_bias_0"T
'while_gru_cell_gru_cell_gru_cell_kernel)while_gru_cell_gru_cell_gru_cell_kernel_0"h
1while_gru_cell_gru_cell_gru_cell_recurrent_kernel3while_gru_cell_gru_cell_gru_cell_recurrent_kernel_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*6
_input_shapes%
#: : : : :	: : :::2P
&while/gru_cell/StatefulPartitionedCall&while/gru_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
: 


B__inference_gru_cell_layer_call_and_return_conditional_losses_4053

inputs
states_0)
%readvariableop_gru_cell_gru_cell_bias2
.matmul_readvariableop_gru_cell_gru_cell_kernel>
:matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel
identity

identity_1
ReadVariableOpReadVariableOp%readvariableop_gru_cell_gru_cell_bias*
_output_shapes
:	*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOp.matmul_readvariableop_gru_cell_gru_cell_kernel* 
_output_shapes
:
ΐ*
dtype02
MatMul/ReadVariableOpk
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
MatMulk
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*
_output_shapes
:	2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split/split_dim
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
split―
MatMul_1/ReadVariableOpReadVariableOp:matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOps
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2

MatMul_1q
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*
_output_shapes
:	2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ????2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split_1/split_dim»
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2	
split_1_
addAddV2split:output:0split_1:output:0*
T0*
_output_shapes
:	2
addP
SigmoidSigmoidadd:z:0*
T0*
_output_shapes
:	2	
Sigmoidc
add_1AddV2split:output:1split_1:output:1*
T0*
_output_shapes
:	2
add_1V
	Sigmoid_1Sigmoid	add_1:z:0*
T0*
_output_shapes
:	2
	Sigmoid_1\
mulMulSigmoid_1:y:0split_1:output:2*
T0*
_output_shapes
:	2
mulZ
add_2AddV2split:output:2mul:z:0*
T0*
_output_shapes
:	2
add_2I
TanhTanh	add_2:z:0*
T0*
_output_shapes
:	2
TanhV
mul_1MulSigmoid:y:0states_0*
T0*
_output_shapes
:	2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xX
subSubsub/x:output:0Sigmoid:y:0*
T0*
_output_shapes
:	2
subR
mul_2Mulsub:z:0Tanh:y:0*
T0*
_output_shapes
:	2
mul_2W
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*
_output_shapes
:	2
add_3U
IdentityIdentity	add_3:z:0*
T0*
_output_shapes
:	2

IdentityY

Identity_1Identity	add_3:z:0*
T0*
_output_shapes
:	2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*5
_input_shapes$
":	ΐ:	::::G C

_output_shapes
:	ΐ
 
_user_specified_nameinputs:IE

_output_shapes
:	
"
_user_specified_name
states/0
ί
Έ
+__inference_functional_1_layer_call_fn_3005
input_1
stream_conv2d_kernel
stream_conv2d_bias
stream_1_conv2d_1_kernel
stream_1_conv2d_1_bias
gru_cell_gru_cell_bias
gru_cell_gru_cell_kernel
gru_cell_variable&
"gru_cell_gru_cell_recurrent_kernel
dense_kernel

dense_bias
dense_1_kernel
dense_1_bias
dense_2_kernel
dense_2_bias
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1stream_conv2d_kernelstream_conv2d_biasstream_1_conv2d_1_kernelstream_1_conv2d_1_biasgru_cell_gru_cell_biasgru_cell_gru_cell_kernelgru_cell_variable"gru_cell_gru_cell_recurrent_kerneldense_kernel
dense_biasdense_1_kerneldense_1_biasdense_2_kerneldense_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*/
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_functional_1_layer_call_and_return_conditional_losses_20532
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:?????????1(::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????1(
!
_user_specified_name	input_1
φ
C
'__inference_stream_2_layer_call_fn_3422

inputs
identityΈ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_stream_2_layer_call_and_return_conditional_losses_18882
PartitionedCalld
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*"
_input_shapes
::K G
#
_output_shapes
:
 
_user_specified_nameinputs
ΒΊ
ϊ
__inference__wrapped_model_877
input_1I
Efunctional_1_stream_conv2d_conv2d_readvariableop_stream_conv2d_kernelH
Dfunctional_1_stream_conv2d_biasadd_readvariableop_stream_conv2d_biasQ
Mfunctional_1_stream_1_conv2d_1_conv2d_readvariableop_stream_1_conv2d_1_kernelP
Lfunctional_1_stream_1_conv2d_1_biasadd_readvariableop_stream_1_conv2d_1_biasH
Dfunctional_1_gru_cell_gru_cell_readvariableop_gru_cell_gru_cell_biasQ
Mfunctional_1_gru_cell_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernelL
Hfunctional_1_gru_cell_gru_cell_matmul_1_readvariableop_gru_cell_variable_
[functional_1_gru_cell_gru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel9
5functional_1_dense_matmul_readvariableop_dense_kernel8
4functional_1_dense_biasadd_readvariableop_dense_bias=
9functional_1_dense_1_matmul_readvariableop_dense_1_kernel<
8functional_1_dense_1_biasadd_readvariableop_dense_1_bias=
9functional_1_dense_2_matmul_readvariableop_dense_2_kernel<
8functional_1_dense_2_biasadd_readvariableop_dense_2_bias
identity’&functional_1/gru/cell/AssignVariableOp’functional_1/gru/cell/while³
2functional_1/tf_op_layer_ExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????24
2functional_1/tf_op_layer_ExpandDims/ExpandDims/dimτ
.functional_1/tf_op_layer_ExpandDims/ExpandDims
ExpandDimsinput_1;functional_1/tf_op_layer_ExpandDims/ExpandDims/dim:output:0*
T0*
_cloned(*&
_output_shapes
:1(20
.functional_1/tf_op_layer_ExpandDims/ExpandDimsς
0functional_1/stream/conv2d/Conv2D/ReadVariableOpReadVariableOpEfunctional_1_stream_conv2d_conv2d_readvariableop_stream_conv2d_kernel*&
_output_shapes
:*
dtype022
0functional_1/stream/conv2d/Conv2D/ReadVariableOp
!functional_1/stream/conv2d/Conv2DConv2D7functional_1/tf_op_layer_ExpandDims/ExpandDims:output:08functional_1/stream/conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:/&*
paddingVALID*
strides
2#
!functional_1/stream/conv2d/Conv2Dη
1functional_1/stream/conv2d/BiasAdd/ReadVariableOpReadVariableOpDfunctional_1_stream_conv2d_biasadd_readvariableop_stream_conv2d_bias*
_output_shapes
:*
dtype023
1functional_1/stream/conv2d/BiasAdd/ReadVariableOpλ
"functional_1/stream/conv2d/BiasAddBiasAdd*functional_1/stream/conv2d/Conv2D:output:09functional_1/stream/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:/&2$
"functional_1/stream/conv2d/BiasAdd¨
functional_1/stream/conv2d/ReluRelu+functional_1/stream/conv2d/BiasAdd:output:0*
T0*&
_output_shapes
:/&2!
functional_1/stream/conv2d/Relu
4functional_1/stream_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOpMfunctional_1_stream_1_conv2d_1_conv2d_readvariableop_stream_1_conv2d_1_kernel*&
_output_shapes
:*
dtype026
4functional_1/stream_1/conv2d_1/Conv2D/ReadVariableOp
%functional_1/stream_1/conv2d_1/Conv2DConv2D-functional_1/stream/conv2d/Relu:activations:0<functional_1/stream_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:+$*
paddingVALID*
strides
2'
%functional_1/stream_1/conv2d_1/Conv2Dχ
5functional_1/stream_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpLfunctional_1_stream_1_conv2d_1_biasadd_readvariableop_stream_1_conv2d_1_bias*
_output_shapes
:*
dtype027
5functional_1/stream_1/conv2d_1/BiasAdd/ReadVariableOpϋ
&functional_1/stream_1/conv2d_1/BiasAddBiasAdd.functional_1/stream_1/conv2d_1/Conv2D:output:0=functional_1/stream_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:+$2(
&functional_1/stream_1/conv2d_1/BiasAdd΄
#functional_1/stream_1/conv2d_1/ReluRelu/functional_1/stream_1/conv2d_1/BiasAdd:output:0*
T0*&
_output_shapes
:+$2%
#functional_1/stream_1/conv2d_1/Relu
functional_1/reshape/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"   +   $      2
functional_1/reshape/Shape
(functional_1/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(functional_1/reshape/strided_slice/stack’
*functional_1/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*functional_1/reshape/strided_slice/stack_1’
*functional_1/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*functional_1/reshape/strided_slice/stack_2ΰ
"functional_1/reshape/strided_sliceStridedSlice#functional_1/reshape/Shape:output:01functional_1/reshape/strided_slice/stack:output:03functional_1/reshape/strided_slice/stack_1:output:03functional_1/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"functional_1/reshape/strided_slice
$functional_1/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$functional_1/reshape/Reshape/shape/1
$functional_1/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :ΐ2&
$functional_1/reshape/Reshape/shape/2
"functional_1/reshape/Reshape/shapePack+functional_1/reshape/strided_slice:output:0-functional_1/reshape/Reshape/shape/1:output:0-functional_1/reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2$
"functional_1/reshape/Reshape/shapeΥ
functional_1/reshape/ReshapeReshape1functional_1/stream_1/conv2d_1/Relu:activations:0+functional_1/reshape/Reshape/shape:output:0*
T0*#
_output_shapes
:+ΐ2
functional_1/reshape/Reshape‘
$functional_1/gru/cell/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$functional_1/gru/cell/transpose/permΣ
functional_1/gru/cell/transpose	Transpose%functional_1/reshape/Reshape:output:0-functional_1/gru/cell/transpose/perm:output:0*
T0*#
_output_shapes
:+ΐ2!
functional_1/gru/cell/transpose
functional_1/gru/cell/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"+      @  2
functional_1/gru/cell/Shape 
)functional_1/gru/cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)functional_1/gru/cell/strided_slice/stack€
+functional_1/gru/cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+functional_1/gru/cell/strided_slice/stack_1€
+functional_1/gru/cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+functional_1/gru/cell/strided_slice/stack_2ζ
#functional_1/gru/cell/strided_sliceStridedSlice$functional_1/gru/cell/Shape:output:02functional_1/gru/cell/strided_slice/stack:output:04functional_1/gru/cell/strided_slice/stack_1:output:04functional_1/gru/cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#functional_1/gru/cell/strided_slice±
1functional_1/gru/cell/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????23
1functional_1/gru/cell/TensorArrayV2/element_shape
#functional_1/gru/cell/TensorArrayV2TensorListReserve:functional_1/gru/cell/TensorArrayV2/element_shape:output:0,functional_1/gru/cell/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#functional_1/gru/cell/TensorArrayV2λ
Kfunctional_1/gru/cell/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   @  2M
Kfunctional_1/gru/cell/TensorArrayUnstack/TensorListFromTensor/element_shapeΠ
=functional_1/gru/cell/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#functional_1/gru/cell/transpose:y:0Tfunctional_1/gru/cell/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=functional_1/gru/cell/TensorArrayUnstack/TensorListFromTensor€
+functional_1/gru/cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+functional_1/gru/cell/strided_slice_1/stack¨
-functional_1/gru/cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-functional_1/gru/cell/strided_slice_1/stack_1¨
-functional_1/gru/cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-functional_1/gru/cell/strided_slice_1/stack_2ψ
%functional_1/gru/cell/strided_slice_1StridedSlice#functional_1/gru/cell/transpose:y:04functional_1/gru/cell/strided_slice_1/stack:output:06functional_1/gru/cell/strided_slice_1/stack_1:output:06functional_1/gru/cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ΐ*
shrink_axis_mask2'
%functional_1/gru/cell/strided_slice_1δ
-functional_1/gru/cell/gru_cell/ReadVariableOpReadVariableOpDfunctional_1_gru_cell_gru_cell_readvariableop_gru_cell_gru_cell_bias*
_output_shapes
:	*
dtype02/
-functional_1/gru/cell/gru_cell/ReadVariableOpΙ
&functional_1/gru/cell/gru_cell/unstackUnpack5functional_1/gru/cell/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2(
&functional_1/gru/cell/gru_cell/unstackό
4functional_1/gru/cell/gru_cell/MatMul/ReadVariableOpReadVariableOpMfunctional_1_gru_cell_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel* 
_output_shapes
:
ΐ*
dtype026
4functional_1/gru/cell/gru_cell/MatMul/ReadVariableOpπ
%functional_1/gru/cell/gru_cell/MatMulMatMul.functional_1/gru/cell/strided_slice_1:output:0<functional_1/gru/cell/gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2'
%functional_1/gru/cell/gru_cell/MatMulη
&functional_1/gru/cell/gru_cell/BiasAddBiasAdd/functional_1/gru/cell/gru_cell/MatMul:product:0/functional_1/gru/cell/gru_cell/unstack:output:0*
T0*
_output_shapes
:	2(
&functional_1/gru/cell/gru_cell/BiasAdd
$functional_1/gru/cell/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2&
$functional_1/gru/cell/gru_cell/Const«
.functional_1/gru/cell/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????20
.functional_1/gru/cell/gru_cell/split/split_dim
$functional_1/gru/cell/gru_cell/splitSplit7functional_1/gru/cell/gru_cell/split/split_dim:output:0/functional_1/gru/cell/gru_cell/BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2&
$functional_1/gru/cell/gru_cell/splitϊ
6functional_1/gru/cell/gru_cell/MatMul_1/ReadVariableOpReadVariableOpHfunctional_1_gru_cell_gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype028
6functional_1/gru/cell/gru_cell/MatMul_1/ReadVariableOp
8functional_1/gru/cell/gru_cell/MatMul_1/ReadVariableOp_1ReadVariableOp[functional_1_gru_cell_gru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel* 
_output_shapes
:
*
dtype02:
8functional_1/gru/cell/gru_cell/MatMul_1/ReadVariableOp_1
'functional_1/gru/cell/gru_cell/MatMul_1MatMul>functional_1/gru/cell/gru_cell/MatMul_1/ReadVariableOp:value:0@functional_1/gru/cell/gru_cell/MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
:	2)
'functional_1/gru/cell/gru_cell/MatMul_1ν
(functional_1/gru/cell/gru_cell/BiasAdd_1BiasAdd1functional_1/gru/cell/gru_cell/MatMul_1:product:0/functional_1/gru/cell/gru_cell/unstack:output:1*
T0*
_output_shapes
:	2*
(functional_1/gru/cell/gru_cell/BiasAdd_1₯
&functional_1/gru/cell/gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ????2(
&functional_1/gru/cell/gru_cell/Const_1―
0functional_1/gru/cell/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????22
0functional_1/gru/cell/gru_cell/split_1/split_dimΦ
&functional_1/gru/cell/gru_cell/split_1SplitV1functional_1/gru/cell/gru_cell/BiasAdd_1:output:0/functional_1/gru/cell/gru_cell/Const_1:output:09functional_1/gru/cell/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2(
&functional_1/gru/cell/gru_cell/split_1Ϋ
"functional_1/gru/cell/gru_cell/addAddV2-functional_1/gru/cell/gru_cell/split:output:0/functional_1/gru/cell/gru_cell/split_1:output:0*
T0*
_output_shapes
:	2$
"functional_1/gru/cell/gru_cell/add­
&functional_1/gru/cell/gru_cell/SigmoidSigmoid&functional_1/gru/cell/gru_cell/add:z:0*
T0*
_output_shapes
:	2(
&functional_1/gru/cell/gru_cell/Sigmoidί
$functional_1/gru/cell/gru_cell/add_1AddV2-functional_1/gru/cell/gru_cell/split:output:1/functional_1/gru/cell/gru_cell/split_1:output:1*
T0*
_output_shapes
:	2&
$functional_1/gru/cell/gru_cell/add_1³
(functional_1/gru/cell/gru_cell/Sigmoid_1Sigmoid(functional_1/gru/cell/gru_cell/add_1:z:0*
T0*
_output_shapes
:	2*
(functional_1/gru/cell/gru_cell/Sigmoid_1Ψ
"functional_1/gru/cell/gru_cell/mulMul,functional_1/gru/cell/gru_cell/Sigmoid_1:y:0/functional_1/gru/cell/gru_cell/split_1:output:2*
T0*
_output_shapes
:	2$
"functional_1/gru/cell/gru_cell/mulΦ
$functional_1/gru/cell/gru_cell/add_2AddV2-functional_1/gru/cell/gru_cell/split:output:2&functional_1/gru/cell/gru_cell/mul:z:0*
T0*
_output_shapes
:	2&
$functional_1/gru/cell/gru_cell/add_2¦
#functional_1/gru/cell/gru_cell/TanhTanh(functional_1/gru/cell/gru_cell/add_2:z:0*
T0*
_output_shapes
:	2%
#functional_1/gru/cell/gru_cell/Tanhτ
3functional_1/gru/cell/gru_cell/mul_1/ReadVariableOpReadVariableOpHfunctional_1_gru_cell_gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype025
3functional_1/gru/cell/gru_cell/mul_1/ReadVariableOpζ
$functional_1/gru/cell/gru_cell/mul_1Mul*functional_1/gru/cell/gru_cell/Sigmoid:y:0;functional_1/gru/cell/gru_cell/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2&
$functional_1/gru/cell/gru_cell/mul_1
$functional_1/gru/cell/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2&
$functional_1/gru/cell/gru_cell/sub/xΤ
"functional_1/gru/cell/gru_cell/subSub-functional_1/gru/cell/gru_cell/sub/x:output:0*functional_1/gru/cell/gru_cell/Sigmoid:y:0*
T0*
_output_shapes
:	2$
"functional_1/gru/cell/gru_cell/subΞ
$functional_1/gru/cell/gru_cell/mul_2Mul&functional_1/gru/cell/gru_cell/sub:z:0'functional_1/gru/cell/gru_cell/Tanh:y:0*
T0*
_output_shapes
:	2&
$functional_1/gru/cell/gru_cell/mul_2Σ
$functional_1/gru/cell/gru_cell/add_3AddV2(functional_1/gru/cell/gru_cell/mul_1:z:0(functional_1/gru/cell/gru_cell/mul_2:z:0*
T0*
_output_shapes
:	2&
$functional_1/gru/cell/gru_cell/add_3»
3functional_1/gru/cell/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      25
3functional_1/gru/cell/TensorArrayV2_1/element_shape
%functional_1/gru/cell/TensorArrayV2_1TensorListReserve<functional_1/gru/cell/TensorArrayV2_1/element_shape:output:0,functional_1/gru/cell/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%functional_1/gru/cell/TensorArrayV2_1z
functional_1/gru/cell/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
functional_1/gru/cell/timeΦ
$functional_1/gru/cell/ReadVariableOpReadVariableOpHfunctional_1_gru_cell_gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype02&
$functional_1/gru/cell/ReadVariableOp«
.functional_1/gru/cell/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????20
.functional_1/gru/cell/while/maximum_iterations
(functional_1/gru/cell/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(functional_1/gru/cell/while/loop_counter?
functional_1/gru/cell/whileWhile1functional_1/gru/cell/while/loop_counter:output:07functional_1/gru/cell/while/maximum_iterations:output:0#functional_1/gru/cell/time:output:0.functional_1/gru/cell/TensorArrayV2_1:handle:0,functional_1/gru/cell/ReadVariableOp:value:0,functional_1/gru/cell/strided_slice:output:0Mfunctional_1/gru/cell/TensorArrayUnstack/TensorListFromTensor:output_handle:0Dfunctional_1_gru_cell_gru_cell_readvariableop_gru_cell_gru_cell_biasMfunctional_1_gru_cell_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel[functional_1_gru_cell_gru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*1
_output_shapes
: : : : :	: : : : : *%
_read_only_resource_inputs
	*0
body(R&
$functional_1_gru_cell_while_body_763*0
cond(R&
$functional_1_gru_cell_while_cond_762*0
output_shapes
: : : : :	: : : : : *
parallel_iterations 2
functional_1/gru/cell/whileα
Ffunctional_1/gru/cell/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2H
Ffunctional_1/gru/cell/TensorArrayV2Stack/TensorListStack/element_shapeΈ
8functional_1/gru/cell/TensorArrayV2Stack/TensorListStackTensorListStack$functional_1/gru/cell/while:output:3Ofunctional_1/gru/cell/TensorArrayV2Stack/TensorListStack/element_shape:output:0*#
_output_shapes
:+*
element_dtype02:
8functional_1/gru/cell/TensorArrayV2Stack/TensorListStack­
+functional_1/gru/cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2-
+functional_1/gru/cell/strided_slice_2/stack¨
-functional_1/gru/cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-functional_1/gru/cell/strided_slice_2/stack_1¨
-functional_1/gru/cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-functional_1/gru/cell/strided_slice_2/stack_2
%functional_1/gru/cell/strided_slice_2StridedSliceAfunctional_1/gru/cell/TensorArrayV2Stack/TensorListStack:tensor:04functional_1/gru/cell/strided_slice_2/stack:output:06functional_1/gru/cell/strided_slice_2/stack_1:output:06functional_1/gru/cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	*
shrink_axis_mask2'
%functional_1/gru/cell/strided_slice_2₯
&functional_1/gru/cell/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&functional_1/gru/cell/transpose_1/permυ
!functional_1/gru/cell/transpose_1	TransposeAfunctional_1/gru/cell/TensorArrayV2Stack/TensorListStack:tensor:0/functional_1/gru/cell/transpose_1/perm:output:0*
T0*#
_output_shapes
:+2#
!functional_1/gru/cell/transpose_1
functional_1/gru/cell/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
functional_1/gru/cell/runtime
&functional_1/gru/cell/AssignVariableOpAssignVariableOpHfunctional_1_gru_cell_gru_cell_matmul_1_readvariableop_gru_cell_variable$functional_1/gru/cell/while:output:4%^functional_1/gru/cell/ReadVariableOp7^functional_1/gru/cell/gru_cell/MatMul_1/ReadVariableOp4^functional_1/gru/cell/gru_cell/mul_1/ReadVariableOp*
_output_shapes
 *
dtype02(
&functional_1/gru/cell/AssignVariableOp
functional_1/gru/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
functional_1/gru/ExpandDims/dimΠ
functional_1/gru/ExpandDims
ExpandDims.functional_1/gru/cell/strided_slice_2:output:0(functional_1/gru/ExpandDims/dim:output:0*
T0*#
_output_shapes
:2
functional_1/gru/ExpandDims
#functional_1/stream_2/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2%
#functional_1/stream_2/flatten/ConstΧ
%functional_1/stream_2/flatten/ReshapeReshape$functional_1/gru/ExpandDims:output:0,functional_1/stream_2/flatten/Const:output:0*
T0*
_output_shapes
:	2'
%functional_1/stream_2/flatten/Reshape€
functional_1/dropout/IdentityIdentity.functional_1/stream_2/flatten/Reshape:output:0*
T0*
_output_shapes
:	2
functional_1/dropout/IdentityΜ
(functional_1/dense/MatMul/ReadVariableOpReadVariableOp5functional_1_dense_matmul_readvariableop_dense_kernel* 
_output_shapes
:
*
dtype02*
(functional_1/dense/MatMul/ReadVariableOpΔ
functional_1/dense/MatMulMatMul&functional_1/dropout/Identity:output:00functional_1/dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
functional_1/dense/MatMulΘ
)functional_1/dense/BiasAdd/ReadVariableOpReadVariableOp4functional_1_dense_biasadd_readvariableop_dense_bias*
_output_shapes	
:*
dtype02+
)functional_1/dense/BiasAdd/ReadVariableOpΕ
functional_1/dense/BiasAddBiasAdd#functional_1/dense/MatMul:product:01functional_1/dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
functional_1/dense/BiasAddΤ
*functional_1/dense_1/MatMul/ReadVariableOpReadVariableOp9functional_1_dense_1_matmul_readvariableop_dense_1_kernel* 
_output_shapes
:
*
dtype02,
*functional_1/dense_1/MatMul/ReadVariableOpΗ
functional_1/dense_1/MatMulMatMul#functional_1/dense/BiasAdd:output:02functional_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
functional_1/dense_1/MatMulΠ
+functional_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp8functional_1_dense_1_biasadd_readvariableop_dense_1_bias*
_output_shapes	
:*
dtype02-
+functional_1/dense_1/BiasAdd/ReadVariableOpΝ
functional_1/dense_1/BiasAddBiasAdd%functional_1/dense_1/MatMul:product:03functional_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
functional_1/dense_1/BiasAdd
functional_1/dense_1/ReluRelu%functional_1/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:	2
functional_1/dense_1/ReluΣ
*functional_1/dense_2/MatMul/ReadVariableOpReadVariableOp9functional_1_dense_2_matmul_readvariableop_dense_2_kernel*
_output_shapes
:	*
dtype02,
*functional_1/dense_2/MatMul/ReadVariableOpΚ
functional_1/dense_2/MatMulMatMul'functional_1/dense_1/Relu:activations:02functional_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/dense_2/MatMulΟ
+functional_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp8functional_1_dense_2_biasadd_readvariableop_dense_2_bias*
_output_shapes
:*
dtype02-
+functional_1/dense_2/BiasAdd/ReadVariableOpΜ
functional_1/dense_2/BiasAddBiasAdd%functional_1/dense_2/MatMul:product:03functional_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/dense_2/BiasAdd·
IdentityIdentity%functional_1/dense_2/BiasAdd:output:0'^functional_1/gru/cell/AssignVariableOp^functional_1/gru/cell/while*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:1(::::::::::::::2P
&functional_1/gru/cell/AssignVariableOp&functional_1/gru/cell/AssignVariableOp2:
functional_1/gru/cell/whilefunctional_1/gru/cell/while:T P
+
_output_shapes
:?????????1(
!
_user_specified_name	input_1
²

while_cond_3560
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice2
.while_while_cond_3560___redundant_placeholder02
.while_while_cond_3560___redundant_placeholder12
.while_while_cond_3560___redundant_placeholder22
.while_while_cond_3560___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*8
_input_shapes'
%: : : : :	: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
:
£K
	
gru_cell_while_body_2666.
*gru_cell_while_gru_cell_while_loop_counter4
0gru_cell_while_gru_cell_while_maximum_iterations
gru_cell_while_placeholder 
gru_cell_while_placeholder_1 
gru_cell_while_placeholder_2+
'gru_cell_while_gru_cell_strided_slice_0i
egru_cell_while_tensorarrayv2read_tensorlistgetitem_gru_cell_tensorarrayunstack_tensorlistfromtensor_0C
?gru_cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0L
Hgru_cell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0X
Tgru_cell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0
gru_cell_while_identity
gru_cell_while_identity_1
gru_cell_while_identity_2
gru_cell_while_identity_3
gru_cell_while_identity_4)
%gru_cell_while_gru_cell_strided_sliceg
cgru_cell_while_tensorarrayv2read_tensorlistgetitem_gru_cell_tensorarrayunstack_tensorlistfromtensorA
=gru_cell_while_gru_cell_readvariableop_gru_cell_gru_cell_biasJ
Fgru_cell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernelV
Rgru_cell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernelΥ
@gru/cell/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   @  2B
@gru/cell/while/TensorArrayV2Read/TensorListGetItem/element_shape
2gru/cell/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemegru_cell_while_tensorarrayv2read_tensorlistgetitem_gru_cell_tensorarrayunstack_tensorlistfromtensor_0gru_cell_while_placeholderIgru/cell/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes
:	ΐ*
element_dtype024
2gru/cell/while/TensorArrayV2Read/TensorListGetItemΡ
&gru/cell/while/gru_cell/ReadVariableOpReadVariableOp?gru_cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0*
_output_shapes
:	*
dtype02(
&gru/cell/while/gru_cell/ReadVariableOp΄
gru/cell/while/gru_cell/unstackUnpack.gru/cell/while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2!
gru/cell/while/gru_cell/unstackι
-gru/cell/while/gru_cell/MatMul/ReadVariableOpReadVariableOpHgru_cell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0* 
_output_shapes
:
ΐ*
dtype02/
-gru/cell/while/gru_cell/MatMul/ReadVariableOpζ
gru/cell/while/gru_cell/MatMulMatMul9gru/cell/while/TensorArrayV2Read/TensorListGetItem:item:05gru/cell/while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2 
gru/cell/while/gru_cell/MatMulΛ
gru/cell/while/gru_cell/BiasAddBiasAdd(gru/cell/while/gru_cell/MatMul:product:0(gru/cell/while/gru_cell/unstack:output:0*
T0*
_output_shapes
:	2!
gru/cell/while/gru_cell/BiasAdd
gru/cell/while/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru/cell/while/gru_cell/Const
'gru/cell/while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'gru/cell/while/gru_cell/split/split_dimτ
gru/cell/while/gru_cell/splitSplit0gru/cell/while/gru_cell/split/split_dim:output:0(gru/cell/while/gru_cell/BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
gru/cell/while/gru_cell/splitω
/gru/cell/while/gru_cell/MatMul_1/ReadVariableOpReadVariableOpTgru_cell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0* 
_output_shapes
:
*
dtype021
/gru/cell/while/gru_cell/MatMul_1/ReadVariableOpΟ
 gru/cell/while/gru_cell/MatMul_1MatMulgru_cell_while_placeholder_27gru/cell/while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2"
 gru/cell/while/gru_cell/MatMul_1Ρ
!gru/cell/while/gru_cell/BiasAdd_1BiasAdd*gru/cell/while/gru_cell/MatMul_1:product:0(gru/cell/while/gru_cell/unstack:output:1*
T0*
_output_shapes
:	2#
!gru/cell/while/gru_cell/BiasAdd_1
gru/cell/while/gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ????2!
gru/cell/while/gru_cell/Const_1‘
)gru/cell/while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2+
)gru/cell/while/gru_cell/split_1/split_dim³
gru/cell/while/gru_cell/split_1SplitV*gru/cell/while/gru_cell/BiasAdd_1:output:0(gru/cell/while/gru_cell/Const_1:output:02gru/cell/while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2!
gru/cell/while/gru_cell/split_1Ώ
gru/cell/while/gru_cell/addAddV2&gru/cell/while/gru_cell/split:output:0(gru/cell/while/gru_cell/split_1:output:0*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/add
gru/cell/while/gru_cell/SigmoidSigmoidgru/cell/while/gru_cell/add:z:0*
T0*
_output_shapes
:	2!
gru/cell/while/gru_cell/SigmoidΓ
gru/cell/while/gru_cell/add_1AddV2&gru/cell/while/gru_cell/split:output:1(gru/cell/while/gru_cell/split_1:output:1*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/add_1
!gru/cell/while/gru_cell/Sigmoid_1Sigmoid!gru/cell/while/gru_cell/add_1:z:0*
T0*
_output_shapes
:	2#
!gru/cell/while/gru_cell/Sigmoid_1Ό
gru/cell/while/gru_cell/mulMul%gru/cell/while/gru_cell/Sigmoid_1:y:0(gru/cell/while/gru_cell/split_1:output:2*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/mulΊ
gru/cell/while/gru_cell/add_2AddV2&gru/cell/while/gru_cell/split:output:2gru/cell/while/gru_cell/mul:z:0*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/add_2
gru/cell/while/gru_cell/TanhTanh!gru/cell/while/gru_cell/add_2:z:0*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/Tanh²
gru/cell/while/gru_cell/mul_1Mul#gru/cell/while/gru_cell/Sigmoid:y:0gru_cell_while_placeholder_2*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/mul_1
gru/cell/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru/cell/while/gru_cell/sub/xΈ
gru/cell/while/gru_cell/subSub&gru/cell/while/gru_cell/sub/x:output:0#gru/cell/while/gru_cell/Sigmoid:y:0*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/sub²
gru/cell/while/gru_cell/mul_2Mulgru/cell/while/gru_cell/sub:z:0 gru/cell/while/gru_cell/Tanh:y:0*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/mul_2·
gru/cell/while/gru_cell/add_3AddV2!gru/cell/while/gru_cell/mul_1:z:0!gru/cell/while/gru_cell/mul_2:z:0*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/add_3
3gru/cell/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_cell_while_placeholder_1gru_cell_while_placeholder!gru/cell/while/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype025
3gru/cell/while/TensorArrayV2Write/TensorListSetItemn
gru/cell/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru/cell/while/add/y
gru/cell/while/addAddV2gru_cell_while_placeholdergru/cell/while/add/y:output:0*
T0*
_output_shapes
: 2
gru/cell/while/addr
gru/cell/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru/cell/while/add_1/y£
gru/cell/while/add_1AddV2*gru_cell_while_gru_cell_while_loop_countergru/cell/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru/cell/while/add_1y
gru/cell/while/IdentityIdentitygru/cell/while/add_1:z:0*
T0*
_output_shapes
: 2
gru/cell/while/Identity
gru/cell/while/Identity_1Identity0gru_cell_while_gru_cell_while_maximum_iterations*
T0*
_output_shapes
: 2
gru/cell/while/Identity_1{
gru/cell/while/Identity_2Identitygru/cell/while/add:z:0*
T0*
_output_shapes
: 2
gru/cell/while/Identity_2¨
gru/cell/while/Identity_3IdentityCgru/cell/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
gru/cell/while/Identity_3
gru/cell/while/Identity_4Identity!gru/cell/while/gru_cell/add_3:z:0*
T0*
_output_shapes
:	2
gru/cell/while/Identity_4"ͺ
Rgru_cell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernelTgru_cell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0"
Fgru_cell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernelHgru_cell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0"
=gru_cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias?gru_cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0"P
%gru_cell_while_gru_cell_strided_slice'gru_cell_while_gru_cell_strided_slice_0";
gru_cell_while_identity gru/cell/while/Identity:output:0"?
gru_cell_while_identity_1"gru/cell/while/Identity_1:output:0"?
gru_cell_while_identity_2"gru/cell/while/Identity_2:output:0"?
gru_cell_while_identity_3"gru/cell/while/Identity_3:output:0"?
gru_cell_while_identity_4"gru/cell/while/Identity_4:output:0"Μ
cgru_cell_while_tensorarrayv2read_tensorlistgetitem_gru_cell_tensorarrayunstack_tensorlistfromtensoregru_cell_while_tensorarrayv2read_tensorlistgetitem_gru_cell_tensorarrayunstack_tensorlistfromtensor_0*6
_input_shapes%
#: : : : :	: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
: 
ΐ
^
B__inference_stream_2_layer_call_and_return_conditional_losses_1888

inputs
identityo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten/Constw
flatten/ReshapeReshapeinputsflatten/Const:output:0*
T0*
_output_shapes
:	2
flatten/Reshaped
IdentityIdentityflatten/Reshape:output:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*"
_input_shapes
::K G
#
_output_shapes
:
 
_user_specified_nameinputs
€
_
A__inference_dropout_layer_call_and_return_conditional_losses_1913

inputs

identity_1R
IdentityIdentityinputs*
T0*
_output_shapes
:	2

Identitya

Identity_1IdentityIdentity:output:0*
T0*
_output_shapes
:	2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs
Ύ	
ΰ
'__inference_gru_cell_layer_call_fn_3959

inputs
states_0
gru_cell_gru_cell_bias
gru_cell_gru_cell_kernel&
"gru_cell_gru_cell_recurrent_kernel
identity

identity_1’StatefulPartitionedCallΈ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0gru_cell_gru_cell_biasgru_cell_gru_cell_kernel"gru_cell_gru_cell_recurrent_kernel*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

::*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_gru_cell_layer_call_and_return_conditional_losses_39512
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*.
_input_shapes
:	ΐ::::22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	ΐ
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0
Ά4
Έ
>__inference_cell_layer_call_and_return_conditional_losses_1461

inputs
gru_cell_gru_cell_variable#
gru_cell_gru_cell_gru_cell_bias%
!gru_cell_gru_cell_gru_cell_kernel/
+gru_cell_gru_cell_gru_cell_recurrent_kernel
identity’AssignVariableOp’ gru_cell/StatefulPartitionedCall’whileu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:?????????ΐ2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2β
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape°
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2Ώ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   @  27
5TensorArrayUnstack/TensorListFromTensor/element_shapeψ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2τ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ΐ*
shrink_axis_mask2
strided_slice_1
 gru_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_1:output:0gru_cell_gru_cell_variablegru_cell_gru_cell_gru_cell_bias!gru_cell_gru_cell_gru_cell_kernel+gru_cell_gru_cell_gru_cell_recurrent_kernel*
Tin	
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:	:	*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_gru_cell_layer_call_and_return_conditional_losses_9812"
 gru_cell/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
TensorArrayV2_1/element_shapeΆ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time|
ReadVariableOpReadVariableOpgru_cell_gru_cell_variable*
_output_shapes
:	*
dtype02
ReadVariableOp
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0ReadVariableOp:value:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_gru_cell_gru_cell_bias!gru_cell_gru_cell_gru_cell_kernel+gru_cell_gru_cell_gru_cell_recurrent_kernel*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*1
_output_shapes
: : : : :	: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_1400*
condR
while_cond_1399*0
output_shapes
: : : : :	: : : : : *
parallel_iterations 2
while΅
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      22
0TensorArrayV2Stack/TensorListStack/element_shapeι
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:?????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¦
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:?????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime»
AssignVariableOpAssignVariableOpgru_cell_gru_cell_variablewhile:output:4^ReadVariableOp!^gru_cell/StatefulPartitionedCall*
_output_shapes
 *
dtype02
AssignVariableOp’
IdentityIdentitystrided_slice_2:output:0^AssignVariableOp!^gru_cell/StatefulPartitionedCall^while*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:?????????ΐ::::2$
AssignVariableOpAssignVariableOp2D
 gru_cell/StatefulPartitionedCall gru_cell/StatefulPartitionedCall2
whilewhile:T P
,
_output_shapes
:?????????ΐ
 
_user_specified_nameinputs

 
B__inference_gru_cell_layer_call_and_return_conditional_losses_3905

inputs
states_0)
%readvariableop_gru_cell_gru_cell_bias2
.matmul_readvariableop_gru_cell_gru_cell_kernel@
<matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel
identity

identity_1
ReadVariableOpReadVariableOp%readvariableop_gru_cell_gru_cell_bias*
_output_shapes
:	*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOp.matmul_readvariableop_gru_cell_gru_cell_kernel* 
_output_shapes
:
ΐ*
dtype02
MatMul/ReadVariableOpk
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
MatMulk
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*
_output_shapes
:	2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split/split_dim
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
splitu
MatMul_1/ReadVariableOpReadVariableOpstates_0*
_output_shapes
:*
dtype02
MatMul_1/ReadVariableOp΅
MatMul_1/ReadVariableOp_1ReadVariableOp<matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOp_1
MatMul_1BatchMatMulV2MatMul_1/ReadVariableOp:value:0!MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
:2

MatMul_1i
	BiasAdd_1BiasAddMatMul_1:output:0unstack:output:1*
T0*
_output_shapes
:2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ????2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split_1/split_dim¦
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0* 
_output_shapes
:::*
	num_split2	
split_1X
addAddV2split:output:0split_1:output:0*
T0*
_output_shapes
:2
addI
SigmoidSigmoidadd:z:0*
T0*
_output_shapes
:2	
Sigmoid\
add_1AddV2split:output:1split_1:output:1*
T0*
_output_shapes
:2
add_1O
	Sigmoid_1Sigmoid	add_1:z:0*
T0*
_output_shapes
:2
	Sigmoid_1U
mulMulSigmoid_1:y:0split_1:output:2*
T0*
_output_shapes
:2
mulS
add_2AddV2split:output:2mul:z:0*
T0*
_output_shapes
:2
add_2B
TanhTanh	add_2:z:0*
T0*
_output_shapes
:2
Tanho
mul_1/ReadVariableOpReadVariableOpstates_0*
_output_shapes
:*
dtype02
mul_1/ReadVariableOpc
mul_1MulSigmoid:y:0mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xQ
subSubsub/x:output:0Sigmoid:y:0*
T0*
_output_shapes
:2
subK
mul_2Mulsub:z:0Tanh:y:0*
T0*
_output_shapes
:2
mul_2P
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*
_output_shapes
:2
add_3N
IdentityIdentity	add_3:z:0*
T0*
_output_shapes
:2

IdentityR

Identity_1Identity	add_3:z:0*
T0*
_output_shapes
:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*.
_input_shapes
:	ΐ:::::G C

_output_shapes
:	ΐ
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0
Έ	
Ι
gru_cell_while_cond_2871.
*gru_cell_while_gru_cell_while_loop_counter4
0gru_cell_while_gru_cell_while_maximum_iterations
gru_cell_while_placeholder 
gru_cell_while_placeholder_1 
gru_cell_while_placeholder_2.
*gru_cell_while_less_gru_cell_strided_sliceD
@gru_cell_while_gru_cell_while_cond_2871___redundant_placeholder0D
@gru_cell_while_gru_cell_while_cond_2871___redundant_placeholder1D
@gru_cell_while_gru_cell_while_cond_2871___redundant_placeholder2D
@gru_cell_while_gru_cell_while_cond_2871___redundant_placeholder3
gru_cell_while_identity

gru/cell/while/LessLessgru_cell_while_placeholder*gru_cell_while_less_gru_cell_strided_slice*
T0*
_output_shapes
: 2
gru/cell/while/Lessx
gru/cell/while/IdentityIdentitygru/cell/while/Less:z:0*
T0
*
_output_shapes
: 2
gru/cell/while/Identity";
gru_cell_while_identity gru/cell/while/Identity:output:0*8
_input_shapes'
%: : : : :	: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
:
ί
Έ
+__inference_functional_1_layer_call_fn_3024
input_1
stream_conv2d_kernel
stream_conv2d_bias
stream_1_conv2d_1_kernel
stream_1_conv2d_1_bias
gru_cell_gru_cell_bias
gru_cell_gru_cell_kernel
gru_cell_variable&
"gru_cell_gru_cell_recurrent_kernel
dense_kernel

dense_bias
dense_1_kernel
dense_1_bias
dense_2_kernel
dense_2_bias
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1stream_conv2d_kernelstream_conv2d_biasstream_1_conv2d_1_kernelstream_1_conv2d_1_biasgru_cell_gru_cell_biasgru_cell_gru_cell_kernelgru_cell_variable"gru_cell_gru_cell_recurrent_kerneldense_kernel
dense_biasdense_1_kerneldense_1_biasdense_2_kerneldense_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*/
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_functional_1_layer_call_and_return_conditional_losses_21002
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:?????????1(::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????1(
!
_user_specified_name	input_1
?
­
?__inference_dense_layer_call_and_return_conditional_losses_3459

inputs&
"matmul_readvariableop_dense_kernel%
!biasadd_readvariableop_dense_bias
identity
MatMul/ReadVariableOpReadVariableOp"matmul_readvariableop_dense_kernel* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpk
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
MatMul
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_dense_bias*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2	
BiasAdd\
IdentityIdentityBiasAdd:output:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*&
_input_shapes
:	:::G C

_output_shapes
:	
 
_user_specified_nameinputs
 
?
F__inference_functional_1_layer_call_and_return_conditional_losses_2787
input_1<
8stream_conv2d_conv2d_readvariableop_stream_conv2d_kernel;
7stream_conv2d_biasadd_readvariableop_stream_conv2d_biasD
@stream_1_conv2d_1_conv2d_readvariableop_stream_1_conv2d_1_kernelC
?stream_1_conv2d_1_biasadd_readvariableop_stream_1_conv2d_1_bias;
7gru_cell_gru_cell_readvariableop_gru_cell_gru_cell_biasD
@gru_cell_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel?
;gru_cell_gru_cell_matmul_1_readvariableop_gru_cell_variableR
Ngru_cell_gru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel,
(dense_matmul_readvariableop_dense_kernel+
'dense_biasadd_readvariableop_dense_bias0
,dense_1_matmul_readvariableop_dense_1_kernel/
+dense_1_biasadd_readvariableop_dense_1_bias0
,dense_2_matmul_readvariableop_dense_2_kernel/
+dense_2_biasadd_readvariableop_dense_2_bias
identity’gru/cell/AssignVariableOp’gru/cell/while
%tf_op_layer_ExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%tf_op_layer_ExpandDims/ExpandDims/dimΝ
!tf_op_layer_ExpandDims/ExpandDims
ExpandDimsinput_1.tf_op_layer_ExpandDims/ExpandDims/dim:output:0*
T0*
_cloned(*&
_output_shapes
:1(2#
!tf_op_layer_ExpandDims/ExpandDimsΛ
#stream/conv2d/Conv2D/ReadVariableOpReadVariableOp8stream_conv2d_conv2d_readvariableop_stream_conv2d_kernel*&
_output_shapes
:*
dtype02%
#stream/conv2d/Conv2D/ReadVariableOpι
stream/conv2d/Conv2DConv2D*tf_op_layer_ExpandDims/ExpandDims:output:0+stream/conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:/&*
paddingVALID*
strides
2
stream/conv2d/Conv2Dΐ
$stream/conv2d/BiasAdd/ReadVariableOpReadVariableOp7stream_conv2d_biasadd_readvariableop_stream_conv2d_bias*
_output_shapes
:*
dtype02&
$stream/conv2d/BiasAdd/ReadVariableOp·
stream/conv2d/BiasAddBiasAddstream/conv2d/Conv2D:output:0,stream/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:/&2
stream/conv2d/BiasAdd
stream/conv2d/ReluRelustream/conv2d/BiasAdd:output:0*
T0*&
_output_shapes
:/&2
stream/conv2d/ReluΫ
'stream_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp@stream_1_conv2d_1_conv2d_readvariableop_stream_1_conv2d_1_kernel*&
_output_shapes
:*
dtype02)
'stream_1/conv2d_1/Conv2D/ReadVariableOpλ
stream_1/conv2d_1/Conv2DConv2D stream/conv2d/Relu:activations:0/stream_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:+$*
paddingVALID*
strides
2
stream_1/conv2d_1/Conv2DΠ
(stream_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp?stream_1_conv2d_1_biasadd_readvariableop_stream_1_conv2d_1_bias*
_output_shapes
:*
dtype02*
(stream_1/conv2d_1/BiasAdd/ReadVariableOpΗ
stream_1/conv2d_1/BiasAddBiasAdd!stream_1/conv2d_1/Conv2D:output:00stream_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:+$2
stream_1/conv2d_1/BiasAdd
stream_1/conv2d_1/ReluRelu"stream_1/conv2d_1/BiasAdd:output:0*
T0*&
_output_shapes
:+$2
stream_1/conv2d_1/Reluw
reshape/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"   +   $      2
reshape/Shape
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slice}
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
reshape/Reshape/shape/1u
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :ΐ2
reshape/Reshape/shape/2Θ
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape‘
reshape/ReshapeReshape$stream_1/conv2d_1/Relu:activations:0reshape/Reshape/shape:output:0*
T0*#
_output_shapes
:+ΐ2
reshape/Reshape
gru/cell/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/cell/transpose/perm
gru/cell/transpose	Transposereshape/Reshape:output:0 gru/cell/transpose/perm:output:0*
T0*#
_output_shapes
:+ΐ2
gru/cell/transposeu
gru/cell/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"+      @  2
gru/cell/Shape
gru/cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/cell/strided_slice/stack
gru/cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru/cell/strided_slice/stack_1
gru/cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru/cell/strided_slice/stack_2
gru/cell/strided_sliceStridedSlicegru/cell/Shape:output:0%gru/cell/strided_slice/stack:output:0'gru/cell/strided_slice/stack_1:output:0'gru/cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru/cell/strided_slice
$gru/cell/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$gru/cell/TensorArrayV2/element_shapeΤ
gru/cell/TensorArrayV2TensorListReserve-gru/cell/TensorArrayV2/element_shape:output:0gru/cell/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru/cell/TensorArrayV2Ρ
>gru/cell/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   @  2@
>gru/cell/TensorArrayUnstack/TensorListFromTensor/element_shape
0gru/cell/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru/cell/transpose:y:0Ggru/cell/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type022
0gru/cell/TensorArrayUnstack/TensorListFromTensor
gru/cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru/cell/strided_slice_1/stack
 gru/cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru/cell/strided_slice_1/stack_1
 gru/cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru/cell/strided_slice_1/stack_2ͺ
gru/cell/strided_slice_1StridedSlicegru/cell/transpose:y:0'gru/cell/strided_slice_1/stack:output:0)gru/cell/strided_slice_1/stack_1:output:0)gru/cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ΐ*
shrink_axis_mask2
gru/cell/strided_slice_1½
 gru/cell/gru_cell/ReadVariableOpReadVariableOp7gru_cell_gru_cell_readvariableop_gru_cell_gru_cell_bias*
_output_shapes
:	*
dtype02"
 gru/cell/gru_cell/ReadVariableOp’
gru/cell/gru_cell/unstackUnpack(gru/cell/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
gru/cell/gru_cell/unstackΥ
'gru/cell/gru_cell/MatMul/ReadVariableOpReadVariableOp@gru_cell_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel* 
_output_shapes
:
ΐ*
dtype02)
'gru/cell/gru_cell/MatMul/ReadVariableOpΌ
gru/cell/gru_cell/MatMulMatMul!gru/cell/strided_slice_1:output:0/gru/cell/gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/MatMul³
gru/cell/gru_cell/BiasAddBiasAdd"gru/cell/gru_cell/MatMul:product:0"gru/cell/gru_cell/unstack:output:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/BiasAddt
gru/cell/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru/cell/gru_cell/Const
!gru/cell/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!gru/cell/gru_cell/split/split_dimά
gru/cell/gru_cell/splitSplit*gru/cell/gru_cell/split/split_dim:output:0"gru/cell/gru_cell/BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
gru/cell/gru_cell/splitΣ
)gru/cell/gru_cell/MatMul_1/ReadVariableOpReadVariableOp;gru_cell_gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype02+
)gru/cell/gru_cell/MatMul_1/ReadVariableOpλ
+gru/cell/gru_cell/MatMul_1/ReadVariableOp_1ReadVariableOpNgru_cell_gru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel* 
_output_shapes
:
*
dtype02-
+gru/cell/gru_cell/MatMul_1/ReadVariableOp_1Τ
gru/cell/gru_cell/MatMul_1MatMul1gru/cell/gru_cell/MatMul_1/ReadVariableOp:value:03gru/cell/gru_cell/MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/MatMul_1Ή
gru/cell/gru_cell/BiasAdd_1BiasAdd$gru/cell/gru_cell/MatMul_1:product:0"gru/cell/gru_cell/unstack:output:1*
T0*
_output_shapes
:	2
gru/cell/gru_cell/BiasAdd_1
gru/cell/gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ????2
gru/cell/gru_cell/Const_1
#gru/cell/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#gru/cell/gru_cell/split_1/split_dim
gru/cell/gru_cell/split_1SplitV$gru/cell/gru_cell/BiasAdd_1:output:0"gru/cell/gru_cell/Const_1:output:0,gru/cell/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2
gru/cell/gru_cell/split_1§
gru/cell/gru_cell/addAddV2 gru/cell/gru_cell/split:output:0"gru/cell/gru_cell/split_1:output:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/add
gru/cell/gru_cell/SigmoidSigmoidgru/cell/gru_cell/add:z:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/Sigmoid«
gru/cell/gru_cell/add_1AddV2 gru/cell/gru_cell/split:output:1"gru/cell/gru_cell/split_1:output:1*
T0*
_output_shapes
:	2
gru/cell/gru_cell/add_1
gru/cell/gru_cell/Sigmoid_1Sigmoidgru/cell/gru_cell/add_1:z:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/Sigmoid_1€
gru/cell/gru_cell/mulMulgru/cell/gru_cell/Sigmoid_1:y:0"gru/cell/gru_cell/split_1:output:2*
T0*
_output_shapes
:	2
gru/cell/gru_cell/mul’
gru/cell/gru_cell/add_2AddV2 gru/cell/gru_cell/split:output:2gru/cell/gru_cell/mul:z:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/add_2
gru/cell/gru_cell/TanhTanhgru/cell/gru_cell/add_2:z:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/TanhΝ
&gru/cell/gru_cell/mul_1/ReadVariableOpReadVariableOp;gru_cell_gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype02(
&gru/cell/gru_cell/mul_1/ReadVariableOp²
gru/cell/gru_cell/mul_1Mulgru/cell/gru_cell/Sigmoid:y:0.gru/cell/gru_cell/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/mul_1w
gru/cell/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru/cell/gru_cell/sub/x 
gru/cell/gru_cell/subSub gru/cell/gru_cell/sub/x:output:0gru/cell/gru_cell/Sigmoid:y:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/sub
gru/cell/gru_cell/mul_2Mulgru/cell/gru_cell/sub:z:0gru/cell/gru_cell/Tanh:y:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/mul_2
gru/cell/gru_cell/add_3AddV2gru/cell/gru_cell/mul_1:z:0gru/cell/gru_cell/mul_2:z:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/add_3‘
&gru/cell/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2(
&gru/cell/TensorArrayV2_1/element_shapeΪ
gru/cell/TensorArrayV2_1TensorListReserve/gru/cell/TensorArrayV2_1/element_shape:output:0gru/cell/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru/cell/TensorArrayV2_1`
gru/cell/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
gru/cell/time―
gru/cell/ReadVariableOpReadVariableOp;gru_cell_gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype02
gru/cell/ReadVariableOp
!gru/cell/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!gru/cell/while/maximum_iterations|
gru/cell/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru/cell/while/loop_counterΛ
gru/cell/whileWhile$gru/cell/while/loop_counter:output:0*gru/cell/while/maximum_iterations:output:0gru/cell/time:output:0!gru/cell/TensorArrayV2_1:handle:0gru/cell/ReadVariableOp:value:0gru/cell/strided_slice:output:0@gru/cell/TensorArrayUnstack/TensorListFromTensor:output_handle:07gru_cell_gru_cell_readvariableop_gru_cell_gru_cell_bias@gru_cell_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernelNgru_cell_gru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*1
_output_shapes
: : : : :	: : : : : *%
_read_only_resource_inputs
	*$
bodyR
gru_cell_while_body_2666*$
condR
gru_cell_while_cond_2665*0
output_shapes
: : : : :	: : : : : *
parallel_iterations 2
gru/cell/whileΗ
9gru/cell/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2;
9gru/cell/TensorArrayV2Stack/TensorListStack/element_shape
+gru/cell/TensorArrayV2Stack/TensorListStackTensorListStackgru/cell/while:output:3Bgru/cell/TensorArrayV2Stack/TensorListStack/element_shape:output:0*#
_output_shapes
:+*
element_dtype02-
+gru/cell/TensorArrayV2Stack/TensorListStack
gru/cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2 
gru/cell/strided_slice_2/stack
 gru/cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru/cell/strided_slice_2/stack_1
 gru/cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru/cell/strided_slice_2/stack_2Θ
gru/cell/strided_slice_2StridedSlice4gru/cell/TensorArrayV2Stack/TensorListStack:tensor:0'gru/cell/strided_slice_2/stack:output:0)gru/cell/strided_slice_2/stack_1:output:0)gru/cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	*
shrink_axis_mask2
gru/cell/strided_slice_2
gru/cell/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/cell/transpose_1/permΑ
gru/cell/transpose_1	Transpose4gru/cell/TensorArrayV2Stack/TensorListStack:tensor:0"gru/cell/transpose_1/perm:output:0*
T0*#
_output_shapes
:+2
gru/cell/transpose_1x
gru/cell/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru/cell/runtime²
gru/cell/AssignVariableOpAssignVariableOp;gru_cell_gru_cell_matmul_1_readvariableop_gru_cell_variablegru/cell/while:output:4^gru/cell/ReadVariableOp*^gru/cell/gru_cell/MatMul_1/ReadVariableOp'^gru/cell/gru_cell/mul_1/ReadVariableOp*
_output_shapes
 *
dtype02
gru/cell/AssignVariableOpj
gru/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
gru/ExpandDims/dim
gru/ExpandDims
ExpandDims!gru/cell/strided_slice_2:output:0gru/ExpandDims/dim:output:0*
T0*#
_output_shapes
:2
gru/ExpandDims
stream_2/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
stream_2/flatten/Const£
stream_2/flatten/ReshapeReshapegru/ExpandDims:output:0stream_2/flatten/Const:output:0*
T0*
_output_shapes
:	2
stream_2/flatten/Reshapes
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *δ8?2
dropout/dropout/Const
dropout/dropout/MulMul!stream_2/flatten/Reshape:output:0dropout/dropout/Const:output:0*
T0*
_output_shapes
:	2
dropout/dropout/Mul
dropout/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2
dropout/dropout/ShapeΔ
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*
_output_shapes
:	*
dtype02.
,dropout/dropout/random_uniform/RandomUniform
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜΜ=2 
dropout/dropout/GreaterEqual/yΦ
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	2
dropout/dropout/GreaterEqual
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	2
dropout/dropout/Cast
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*
_output_shapes
:	2
dropout/dropout/Mul_1₯
dense/MatMul/ReadVariableOpReadVariableOp(dense_matmul_readvariableop_dense_kernel* 
_output_shapes
:
*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMuldropout/dropout/Mul_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense/MatMul‘
dense/BiasAdd/ReadVariableOpReadVariableOp'dense_biasadd_readvariableop_dense_bias*
_output_shapes	
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense/BiasAdd­
dense_1/MatMul/ReadVariableOpReadVariableOp,dense_1_matmul_readvariableop_dense_1_kernel* 
_output_shapes
:
*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldense/BiasAdd:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense_1/MatMul©
dense_1/BiasAdd/ReadVariableOpReadVariableOp+dense_1_biasadd_readvariableop_dense_1_bias*
_output_shapes	
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense_1/BiasAddh
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*
_output_shapes
:	2
dense_1/Relu¬
dense_2/MatMul/ReadVariableOpReadVariableOp,dense_2_matmul_readvariableop_dense_2_kernel*
_output_shapes
:	*
dtype02
dense_2/MatMul/ReadVariableOp
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_2/MatMul¨
dense_2/BiasAdd/ReadVariableOpReadVariableOp+dense_2_biasadd_readvariableop_dense_2_bias*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_2/BiasAdd
IdentityIdentitydense_2/BiasAdd:output:0^gru/cell/AssignVariableOp^gru/cell/while*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:1(::::::::::::::26
gru/cell/AssignVariableOpgru/cell/AssignVariableOp2 
gru/cell/whilegru/cell/while:T P
+
_output_shapes
:?????????1(
!
_user_specified_name	input_1
τP
ξ
>__inference_cell_layer_call_and_return_conditional_losses_3651
inputs_02
.gru_cell_readvariableop_gru_cell_gru_cell_bias;
7gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel6
2gru_cell_matmul_1_readvariableop_gru_cell_variableI
Egru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel
identity’AssignVariableOp’whileu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm}
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*,
_output_shapes
:?????????ΐ2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2β
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape°
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2Ώ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   @  27
5TensorArrayUnstack/TensorListFromTensor/element_shapeψ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2τ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ΐ*
shrink_axis_mask2
strided_slice_1’
gru_cell/ReadVariableOpReadVariableOp.gru_cell_readvariableop_gru_cell_gru_cell_bias*
_output_shapes
:	*
dtype02
gru_cell/ReadVariableOp
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
gru_cell/unstackΊ
gru_cell/MatMul/ReadVariableOpReadVariableOp7gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel* 
_output_shapes
:
ΐ*
dtype02 
gru_cell/MatMul/ReadVariableOp
gru_cell/MatMulMatMulstrided_slice_1:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
gru_cell/MatMul
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*
_output_shapes
:	2
gru_cell/BiasAddb
gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell/Const
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell/split/split_dimΈ
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
gru_cell/splitΈ
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp2gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype02"
 gru_cell/MatMul_1/ReadVariableOpΠ
"gru_cell/MatMul_1/ReadVariableOp_1ReadVariableOpEgru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel* 
_output_shapes
:
*
dtype02$
"gru_cell/MatMul_1/ReadVariableOp_1°
gru_cell/MatMul_1MatMul(gru_cell/MatMul_1/ReadVariableOp:value:0*gru_cell/MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
:	2
gru_cell/MatMul_1
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*
_output_shapes
:	2
gru_cell/BiasAdd_1y
gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ????2
gru_cell/Const_1
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell/split_1/split_dimθ
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const_1:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2
gru_cell/split_1
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*
_output_shapes
:	2
gru_cell/addk
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*
_output_shapes
:	2
gru_cell/Sigmoid
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*
_output_shapes
:	2
gru_cell/add_1q
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*
_output_shapes
:	2
gru_cell/Sigmoid_1
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*
_output_shapes
:	2
gru_cell/mul~
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*
_output_shapes
:	2
gru_cell/add_2d
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*
_output_shapes
:	2
gru_cell/Tanh²
gru_cell/mul_1/ReadVariableOpReadVariableOp2gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype02
gru_cell/mul_1/ReadVariableOp
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0%gru_cell/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
gru_cell/mul_1e
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell/sub/x|
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*
_output_shapes
:	2
gru_cell/subv
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*
_output_shapes
:	2
gru_cell/mul_2{
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*
_output_shapes
:	2
gru_cell/add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
TensorArrayV2_1/element_shapeΆ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
ReadVariableOpReadVariableOp2gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype02
ReadVariableOp
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterΝ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0ReadVariableOp:value:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0.gru_cell_readvariableop_gru_cell_gru_cell_bias7gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernelEgru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*1
_output_shapes
: : : : :	: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_3561*
condR
while_cond_3560*0
output_shapes
: : : : :	: : : : : *
parallel_iterations 2
while΅
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      22
0TensorArrayV2Stack/TensorListStack/element_shapeι
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:?????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¦
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:?????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeσ
AssignVariableOpAssignVariableOp2gru_cell_matmul_1_readvariableop_gru_cell_variablewhile:output:4^ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/mul_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOp
IdentityIdentitystrided_slice_2:output:0^AssignVariableOp^while*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:?????????ΐ::::2$
AssignVariableOpAssignVariableOp2
whilewhile:V R
,
_output_shapes
:?????????ΐ
"
_user_specified_name
inputs/0
?

B__inference_gru_cell_layer_call_and_return_conditional_losses_4005

inputs

states)
%readvariableop_gru_cell_gru_cell_bias2
.matmul_readvariableop_gru_cell_gru_cell_kernel@
<matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel
identity

identity_1
ReadVariableOpReadVariableOp%readvariableop_gru_cell_gru_cell_bias*
_output_shapes
:	*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOp.matmul_readvariableop_gru_cell_gru_cell_kernel* 
_output_shapes
:
ΐ*
dtype02
MatMul/ReadVariableOpk
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
MatMulk
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*
_output_shapes
:	2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split/split_dim
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
splits
MatMul_1/ReadVariableOpReadVariableOpstates*
_output_shapes
:*
dtype02
MatMul_1/ReadVariableOp΅
MatMul_1/ReadVariableOp_1ReadVariableOp<matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOp_1
MatMul_1BatchMatMulV2MatMul_1/ReadVariableOp:value:0!MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
:2

MatMul_1i
	BiasAdd_1BiasAddMatMul_1:output:0unstack:output:1*
T0*
_output_shapes
:2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ????2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split_1/split_dim¦
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0* 
_output_shapes
:::*
	num_split2	
split_1X
addAddV2split:output:0split_1:output:0*
T0*
_output_shapes
:2
addI
SigmoidSigmoidadd:z:0*
T0*
_output_shapes
:2	
Sigmoid\
add_1AddV2split:output:1split_1:output:1*
T0*
_output_shapes
:2
add_1O
	Sigmoid_1Sigmoid	add_1:z:0*
T0*
_output_shapes
:2
	Sigmoid_1U
mulMulSigmoid_1:y:0split_1:output:2*
T0*
_output_shapes
:2
mulS
add_2AddV2split:output:2mul:z:0*
T0*
_output_shapes
:2
add_2B
TanhTanh	add_2:z:0*
T0*
_output_shapes
:2
Tanhm
mul_1/ReadVariableOpReadVariableOpstates*
_output_shapes
:*
dtype02
mul_1/ReadVariableOpc
mul_1MulSigmoid:y:0mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xQ
subSubsub/x:output:0Sigmoid:y:0*
T0*
_output_shapes
:2
subK
mul_2Mulsub:z:0Tanh:y:0*
T0*
_output_shapes
:2
mul_2P
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*
_output_shapes
:2
add_3N
IdentityIdentity	add_3:z:0*
T0*
_output_shapes
:2

IdentityR

Identity_1Identity	add_3:z:0*
T0*
_output_shapes
:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*.
_input_shapes
:	ΐ:::::G C

_output_shapes
:	ΐ
 
_user_specified_nameinputs:&"
 
_user_specified_namestates


B__inference_gru_cell_layer_call_and_return_conditional_losses_1077

inputs

states)
%readvariableop_gru_cell_gru_cell_bias2
.matmul_readvariableop_gru_cell_gru_cell_kernel>
:matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel
identity

identity_1
ReadVariableOpReadVariableOp%readvariableop_gru_cell_gru_cell_bias*
_output_shapes
:	*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOp.matmul_readvariableop_gru_cell_gru_cell_kernel* 
_output_shapes
:
ΐ*
dtype02
MatMul/ReadVariableOpk
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
MatMulk
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*
_output_shapes
:	2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split/split_dim
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
split―
MatMul_1/ReadVariableOpReadVariableOp:matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOpq
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2

MatMul_1q
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*
_output_shapes
:	2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ????2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split_1/split_dim»
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2	
split_1_
addAddV2split:output:0split_1:output:0*
T0*
_output_shapes
:	2
addP
SigmoidSigmoidadd:z:0*
T0*
_output_shapes
:	2	
Sigmoidc
add_1AddV2split:output:1split_1:output:1*
T0*
_output_shapes
:	2
add_1V
	Sigmoid_1Sigmoid	add_1:z:0*
T0*
_output_shapes
:	2
	Sigmoid_1\
mulMulSigmoid_1:y:0split_1:output:2*
T0*
_output_shapes
:	2
mulZ
add_2AddV2split:output:2mul:z:0*
T0*
_output_shapes
:	2
add_2I
TanhTanh	add_2:z:0*
T0*
_output_shapes
:	2
TanhT
mul_1MulSigmoid:y:0states*
T0*
_output_shapes
:	2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xX
subSubsub/x:output:0Sigmoid:y:0*
T0*
_output_shapes
:	2
subR
mul_2Mulsub:z:0Tanh:y:0*
T0*
_output_shapes
:	2
mul_2W
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*
_output_shapes
:	2
add_3U
IdentityIdentity	add_3:z:0*
T0*
_output_shapes
:	2

IdentityY

Identity_1Identity	add_3:z:0*
T0*
_output_shapes
:	2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*5
_input_shapes$
":	ΐ:	::::G C

_output_shapes
:	ΐ
 
_user_specified_nameinputs:GC

_output_shapes
:	
 
_user_specified_namestates

Χ
#__inference_cell_layer_call_fn_3819
inputs_0
gru_cell_variable
gru_cell_gru_cell_bias
gru_cell_gru_cell_kernel&
"gru_cell_gru_cell_recurrent_kernel
identity’StatefulPartitionedCallΐ
StatefulPartitionedCallStatefulPartitionedCallinputs_0gru_cell_variablegru_cell_gru_cell_biasgru_cell_gru_cell_kernel"gru_cell_gru_cell_recurrent_kernel*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_cell_layer_call_and_return_conditional_losses_14612
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:?????????ΐ::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:?????????ΐ
"
_user_specified_name
inputs/0


ΰ
'__inference_gru_cell_layer_call_fn_4115

inputs
states_0
gru_cell_gru_cell_bias
gru_cell_gru_cell_kernel&
"gru_cell_gru_cell_recurrent_kernel
identity

identity_1’StatefulPartitionedCallΕ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0gru_cell_gru_cell_biasgru_cell_gru_cell_kernel"gru_cell_gru_cell_recurrent_kernel*
Tin	
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:	:	*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_gru_cell_layer_call_and_return_conditional_losses_11172
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes
:	2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*5
_input_shapes$
":	ΐ:	:::22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	ΐ
 
_user_specified_nameinputs:IE

_output_shapes
:	
"
_user_specified_name
states/0
ΛE

cell_while_body_3301&
"cell_while_cell_while_loop_counter,
(cell_while_cell_while_maximum_iterations
cell_while_placeholder
cell_while_placeholder_1
cell_while_placeholder_2#
cell_while_cell_strided_slice_0a
]cell_while_tensorarrayv2read_tensorlistgetitem_cell_tensorarrayunstack_tensorlistfromtensor_0?
;cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0H
Dcell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0T
Pcell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0
cell_while_identity
cell_while_identity_1
cell_while_identity_2
cell_while_identity_3
cell_while_identity_4!
cell_while_cell_strided_slice_
[cell_while_tensorarrayv2read_tensorlistgetitem_cell_tensorarrayunstack_tensorlistfromtensor=
9cell_while_gru_cell_readvariableop_gru_cell_gru_cell_biasF
Bcell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernelR
Ncell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernelΝ
<cell/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   @  2>
<cell/while/TensorArrayV2Read/TensorListGetItem/element_shapeι
.cell/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem]cell_while_tensorarrayv2read_tensorlistgetitem_cell_tensorarrayunstack_tensorlistfromtensor_0cell_while_placeholderEcell/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes
:	ΐ*
element_dtype020
.cell/while/TensorArrayV2Read/TensorListGetItemΕ
"cell/while/gru_cell/ReadVariableOpReadVariableOp;cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0*
_output_shapes
:	*
dtype02$
"cell/while/gru_cell/ReadVariableOp¨
cell/while/gru_cell/unstackUnpack*cell/while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
cell/while/gru_cell/unstackέ
)cell/while/gru_cell/MatMul/ReadVariableOpReadVariableOpDcell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0* 
_output_shapes
:
ΐ*
dtype02+
)cell/while/gru_cell/MatMul/ReadVariableOpΦ
cell/while/gru_cell/MatMulMatMul5cell/while/TensorArrayV2Read/TensorListGetItem:item:01cell/while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/MatMul»
cell/while/gru_cell/BiasAddBiasAdd$cell/while/gru_cell/MatMul:product:0$cell/while/gru_cell/unstack:output:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/BiasAddx
cell/while/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
cell/while/gru_cell/Const
#cell/while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#cell/while/gru_cell/split/split_dimδ
cell/while/gru_cell/splitSplit,cell/while/gru_cell/split/split_dim:output:0$cell/while/gru_cell/BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
cell/while/gru_cell/splitν
+cell/while/gru_cell/MatMul_1/ReadVariableOpReadVariableOpPcell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0* 
_output_shapes
:
*
dtype02-
+cell/while/gru_cell/MatMul_1/ReadVariableOpΏ
cell/while/gru_cell/MatMul_1MatMulcell_while_placeholder_23cell/while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/MatMul_1Α
cell/while/gru_cell/BiasAdd_1BiasAdd&cell/while/gru_cell/MatMul_1:product:0$cell/while/gru_cell/unstack:output:1*
T0*
_output_shapes
:	2
cell/while/gru_cell/BiasAdd_1
cell/while/gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ????2
cell/while/gru_cell/Const_1
%cell/while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%cell/while/gru_cell/split_1/split_dim
cell/while/gru_cell/split_1SplitV&cell/while/gru_cell/BiasAdd_1:output:0$cell/while/gru_cell/Const_1:output:0.cell/while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2
cell/while/gru_cell/split_1―
cell/while/gru_cell/addAddV2"cell/while/gru_cell/split:output:0$cell/while/gru_cell/split_1:output:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/add
cell/while/gru_cell/SigmoidSigmoidcell/while/gru_cell/add:z:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/Sigmoid³
cell/while/gru_cell/add_1AddV2"cell/while/gru_cell/split:output:1$cell/while/gru_cell/split_1:output:1*
T0*
_output_shapes
:	2
cell/while/gru_cell/add_1
cell/while/gru_cell/Sigmoid_1Sigmoidcell/while/gru_cell/add_1:z:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/Sigmoid_1¬
cell/while/gru_cell/mulMul!cell/while/gru_cell/Sigmoid_1:y:0$cell/while/gru_cell/split_1:output:2*
T0*
_output_shapes
:	2
cell/while/gru_cell/mulͺ
cell/while/gru_cell/add_2AddV2"cell/while/gru_cell/split:output:2cell/while/gru_cell/mul:z:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/add_2
cell/while/gru_cell/TanhTanhcell/while/gru_cell/add_2:z:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/Tanh’
cell/while/gru_cell/mul_1Mulcell/while/gru_cell/Sigmoid:y:0cell_while_placeholder_2*
T0*
_output_shapes
:	2
cell/while/gru_cell/mul_1{
cell/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
cell/while/gru_cell/sub/x¨
cell/while/gru_cell/subSub"cell/while/gru_cell/sub/x:output:0cell/while/gru_cell/Sigmoid:y:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/sub’
cell/while/gru_cell/mul_2Mulcell/while/gru_cell/sub:z:0cell/while/gru_cell/Tanh:y:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/mul_2§
cell/while/gru_cell/add_3AddV2cell/while/gru_cell/mul_1:z:0cell/while/gru_cell/mul_2:z:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/add_3υ
/cell/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemcell_while_placeholder_1cell_while_placeholdercell/while/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype021
/cell/while/TensorArrayV2Write/TensorListSetItemf
cell/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
cell/while/add/y}
cell/while/addAddV2cell_while_placeholdercell/while/add/y:output:0*
T0*
_output_shapes
: 2
cell/while/addj
cell/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
cell/while/add_1/y
cell/while/add_1AddV2"cell_while_cell_while_loop_countercell/while/add_1/y:output:0*
T0*
_output_shapes
: 2
cell/while/add_1m
cell/while/IdentityIdentitycell/while/add_1:z:0*
T0*
_output_shapes
: 2
cell/while/Identity
cell/while/Identity_1Identity(cell_while_cell_while_maximum_iterations*
T0*
_output_shapes
: 2
cell/while/Identity_1o
cell/while/Identity_2Identitycell/while/add:z:0*
T0*
_output_shapes
: 2
cell/while/Identity_2
cell/while/Identity_3Identity?cell/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
cell/while/Identity_3
cell/while/Identity_4Identitycell/while/gru_cell/add_3:z:0*
T0*
_output_shapes
:	2
cell/while/Identity_4"@
cell_while_cell_strided_slicecell_while_cell_strided_slice_0"’
Ncell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernelPcell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0"
Bcell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernelDcell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0"x
9cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias;cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0"3
cell_while_identitycell/while/Identity:output:0"7
cell_while_identity_1cell/while/Identity_1:output:0"7
cell_while_identity_2cell/while/Identity_2:output:0"7
cell_while_identity_3cell/while/Identity_3:output:0"7
cell_while_identity_4cell/while/Identity_4:output:0"Ό
[cell_while_tensorarrayv2read_tensorlistgetitem_cell_tensorarrayunstack_tensorlistfromtensor]cell_while_tensorarrayv2read_tensorlistgetitem_cell_tensorarrayunstack_tensorlistfromtensor_0*6
_input_shapes%
#: : : : :	: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
: 
υ
]
A__inference_reshape_layer_call_and_return_conditional_losses_1545

inputs
identityg
ShapeConst*
_output_shapes
:*
dtype0*%
valueB"   +   $      2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2β
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicem
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape/shape/1e
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :ΐ2
Reshape/shape/2 
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapek
ReshapeReshapeinputsReshape/shape:output:0*
T0*#
_output_shapes
:+ΐ2	
Reshape`
IdentityIdentityReshape:output:0*
T0*#
_output_shapes
:+ΐ2

Identity"
identityIdentity:output:0*%
_input_shapes
:+$:N J
&
_output_shapes
:+$
 
_user_specified_nameinputs
Έ	
Ι
gru_cell_while_cond_2428.
*gru_cell_while_gru_cell_while_loop_counter4
0gru_cell_while_gru_cell_while_maximum_iterations
gru_cell_while_placeholder 
gru_cell_while_placeholder_1 
gru_cell_while_placeholder_2.
*gru_cell_while_less_gru_cell_strided_sliceD
@gru_cell_while_gru_cell_while_cond_2428___redundant_placeholder0D
@gru_cell_while_gru_cell_while_cond_2428___redundant_placeholder1D
@gru_cell_while_gru_cell_while_cond_2428___redundant_placeholder2D
@gru_cell_while_gru_cell_while_cond_2428___redundant_placeholder3
gru_cell_while_identity

gru/cell/while/LessLessgru_cell_while_placeholder*gru_cell_while_less_gru_cell_strided_slice*
T0*
_output_shapes
: 2
gru/cell/while/Lessx
gru/cell/while/IdentityIdentitygru/cell/while/Less:z:0*
T0
*
_output_shapes
: 2
gru/cell/while/Identity";
gru_cell_while_identity gru/cell/while/Identity:output:0*8
_input_shapes'
%: : : : :	: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
:
Ϋ
l
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_3030

inputs
identityk
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims/dim

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*
_cloned(*&
_output_shapes
:1(2

ExpandDimsf
IdentityIdentityExpandDims:output:0*
T0*&
_output_shapes
:1(2

Identity"
identityIdentity:output:0*!
_input_shapes
:1(:J F
"
_output_shapes
:1(
 
_user_specified_nameinputs
ά
·
+__inference_functional_1_layer_call_fn_2581

inputs
stream_conv2d_kernel
stream_conv2d_bias
stream_1_conv2d_1_kernel
stream_1_conv2d_1_bias
gru_cell_gru_cell_bias
gru_cell_gru_cell_kernel
gru_cell_variable&
"gru_cell_gru_cell_recurrent_kernel
dense_kernel

dense_bias
dense_1_kernel
dense_1_bias
dense_2_kernel
dense_2_bias
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstream_conv2d_kernelstream_conv2d_biasstream_1_conv2d_1_kernelstream_1_conv2d_1_biasgru_cell_gru_cell_biasgru_cell_gru_cell_kernelgru_cell_variable"gru_cell_gru_cell_recurrent_kerneldense_kernel
dense_biasdense_1_kerneldense_1_biasdense_2_kerneldense_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*/
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_functional_1_layer_call_and_return_conditional_losses_21002
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:?????????1(::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????1(
 
_user_specified_nameinputs


B__inference_gru_cell_layer_call_and_return_conditional_losses_1117

inputs

states)
%readvariableop_gru_cell_gru_cell_bias2
.matmul_readvariableop_gru_cell_gru_cell_kernel>
:matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel
identity

identity_1
ReadVariableOpReadVariableOp%readvariableop_gru_cell_gru_cell_bias*
_output_shapes
:	*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOp.matmul_readvariableop_gru_cell_gru_cell_kernel* 
_output_shapes
:
ΐ*
dtype02
MatMul/ReadVariableOpk
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
MatMulk
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*
_output_shapes
:	2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split/split_dim
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
split―
MatMul_1/ReadVariableOpReadVariableOp:matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOpq
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2

MatMul_1q
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*
_output_shapes
:	2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ????2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split_1/split_dim»
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2	
split_1_
addAddV2split:output:0split_1:output:0*
T0*
_output_shapes
:	2
addP
SigmoidSigmoidadd:z:0*
T0*
_output_shapes
:	2	
Sigmoidc
add_1AddV2split:output:1split_1:output:1*
T0*
_output_shapes
:	2
add_1V
	Sigmoid_1Sigmoid	add_1:z:0*
T0*
_output_shapes
:	2
	Sigmoid_1\
mulMulSigmoid_1:y:0split_1:output:2*
T0*
_output_shapes
:	2
mulZ
add_2AddV2split:output:2mul:z:0*
T0*
_output_shapes
:	2
add_2I
TanhTanh	add_2:z:0*
T0*
_output_shapes
:	2
TanhT
mul_1MulSigmoid:y:0states*
T0*
_output_shapes
:	2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xX
subSubsub/x:output:0Sigmoid:y:0*
T0*
_output_shapes
:	2
subR
mul_2Mulsub:z:0Tanh:y:0*
T0*
_output_shapes
:	2
mul_2W
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*
_output_shapes
:	2
add_3U
IdentityIdentity	add_3:z:0*
T0*
_output_shapes
:	2

IdentityY

Identity_1Identity	add_3:z:0*
T0*
_output_shapes
:	2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*5
_input_shapes$
":	ΐ:	::::G C

_output_shapes
:	ΐ
 
_user_specified_nameinputs:GC

_output_shapes
:	
 
_user_specified_namestates
ΛE

cell_while_body_3149&
"cell_while_cell_while_loop_counter,
(cell_while_cell_while_maximum_iterations
cell_while_placeholder
cell_while_placeholder_1
cell_while_placeholder_2#
cell_while_cell_strided_slice_0a
]cell_while_tensorarrayv2read_tensorlistgetitem_cell_tensorarrayunstack_tensorlistfromtensor_0?
;cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0H
Dcell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0T
Pcell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0
cell_while_identity
cell_while_identity_1
cell_while_identity_2
cell_while_identity_3
cell_while_identity_4!
cell_while_cell_strided_slice_
[cell_while_tensorarrayv2read_tensorlistgetitem_cell_tensorarrayunstack_tensorlistfromtensor=
9cell_while_gru_cell_readvariableop_gru_cell_gru_cell_biasF
Bcell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernelR
Ncell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernelΝ
<cell/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   @  2>
<cell/while/TensorArrayV2Read/TensorListGetItem/element_shapeι
.cell/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem]cell_while_tensorarrayv2read_tensorlistgetitem_cell_tensorarrayunstack_tensorlistfromtensor_0cell_while_placeholderEcell/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes
:	ΐ*
element_dtype020
.cell/while/TensorArrayV2Read/TensorListGetItemΕ
"cell/while/gru_cell/ReadVariableOpReadVariableOp;cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0*
_output_shapes
:	*
dtype02$
"cell/while/gru_cell/ReadVariableOp¨
cell/while/gru_cell/unstackUnpack*cell/while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
cell/while/gru_cell/unstackέ
)cell/while/gru_cell/MatMul/ReadVariableOpReadVariableOpDcell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0* 
_output_shapes
:
ΐ*
dtype02+
)cell/while/gru_cell/MatMul/ReadVariableOpΦ
cell/while/gru_cell/MatMulMatMul5cell/while/TensorArrayV2Read/TensorListGetItem:item:01cell/while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/MatMul»
cell/while/gru_cell/BiasAddBiasAdd$cell/while/gru_cell/MatMul:product:0$cell/while/gru_cell/unstack:output:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/BiasAddx
cell/while/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
cell/while/gru_cell/Const
#cell/while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#cell/while/gru_cell/split/split_dimδ
cell/while/gru_cell/splitSplit,cell/while/gru_cell/split/split_dim:output:0$cell/while/gru_cell/BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
cell/while/gru_cell/splitν
+cell/while/gru_cell/MatMul_1/ReadVariableOpReadVariableOpPcell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0* 
_output_shapes
:
*
dtype02-
+cell/while/gru_cell/MatMul_1/ReadVariableOpΏ
cell/while/gru_cell/MatMul_1MatMulcell_while_placeholder_23cell/while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/MatMul_1Α
cell/while/gru_cell/BiasAdd_1BiasAdd&cell/while/gru_cell/MatMul_1:product:0$cell/while/gru_cell/unstack:output:1*
T0*
_output_shapes
:	2
cell/while/gru_cell/BiasAdd_1
cell/while/gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ????2
cell/while/gru_cell/Const_1
%cell/while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%cell/while/gru_cell/split_1/split_dim
cell/while/gru_cell/split_1SplitV&cell/while/gru_cell/BiasAdd_1:output:0$cell/while/gru_cell/Const_1:output:0.cell/while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2
cell/while/gru_cell/split_1―
cell/while/gru_cell/addAddV2"cell/while/gru_cell/split:output:0$cell/while/gru_cell/split_1:output:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/add
cell/while/gru_cell/SigmoidSigmoidcell/while/gru_cell/add:z:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/Sigmoid³
cell/while/gru_cell/add_1AddV2"cell/while/gru_cell/split:output:1$cell/while/gru_cell/split_1:output:1*
T0*
_output_shapes
:	2
cell/while/gru_cell/add_1
cell/while/gru_cell/Sigmoid_1Sigmoidcell/while/gru_cell/add_1:z:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/Sigmoid_1¬
cell/while/gru_cell/mulMul!cell/while/gru_cell/Sigmoid_1:y:0$cell/while/gru_cell/split_1:output:2*
T0*
_output_shapes
:	2
cell/while/gru_cell/mulͺ
cell/while/gru_cell/add_2AddV2"cell/while/gru_cell/split:output:2cell/while/gru_cell/mul:z:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/add_2
cell/while/gru_cell/TanhTanhcell/while/gru_cell/add_2:z:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/Tanh’
cell/while/gru_cell/mul_1Mulcell/while/gru_cell/Sigmoid:y:0cell_while_placeholder_2*
T0*
_output_shapes
:	2
cell/while/gru_cell/mul_1{
cell/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
cell/while/gru_cell/sub/x¨
cell/while/gru_cell/subSub"cell/while/gru_cell/sub/x:output:0cell/while/gru_cell/Sigmoid:y:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/sub’
cell/while/gru_cell/mul_2Mulcell/while/gru_cell/sub:z:0cell/while/gru_cell/Tanh:y:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/mul_2§
cell/while/gru_cell/add_3AddV2cell/while/gru_cell/mul_1:z:0cell/while/gru_cell/mul_2:z:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/add_3υ
/cell/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemcell_while_placeholder_1cell_while_placeholdercell/while/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype021
/cell/while/TensorArrayV2Write/TensorListSetItemf
cell/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
cell/while/add/y}
cell/while/addAddV2cell_while_placeholdercell/while/add/y:output:0*
T0*
_output_shapes
: 2
cell/while/addj
cell/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
cell/while/add_1/y
cell/while/add_1AddV2"cell_while_cell_while_loop_countercell/while/add_1/y:output:0*
T0*
_output_shapes
: 2
cell/while/add_1m
cell/while/IdentityIdentitycell/while/add_1:z:0*
T0*
_output_shapes
: 2
cell/while/Identity
cell/while/Identity_1Identity(cell_while_cell_while_maximum_iterations*
T0*
_output_shapes
: 2
cell/while/Identity_1o
cell/while/Identity_2Identitycell/while/add:z:0*
T0*
_output_shapes
: 2
cell/while/Identity_2
cell/while/Identity_3Identity?cell/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
cell/while/Identity_3
cell/while/Identity_4Identitycell/while/gru_cell/add_3:z:0*
T0*
_output_shapes
:	2
cell/while/Identity_4"@
cell_while_cell_strided_slicecell_while_cell_strided_slice_0"’
Ncell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernelPcell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0"
Bcell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernelDcell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0"x
9cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias;cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0"3
cell_while_identitycell/while/Identity:output:0"7
cell_while_identity_1cell/while/Identity_1:output:0"7
cell_while_identity_2cell/while/Identity_2:output:0"7
cell_while_identity_3cell/while/Identity_3:output:0"7
cell_while_identity_4cell/while/Identity_4:output:0"Ό
[cell_while_tensorarrayv2read_tensorlistgetitem_cell_tensorarrayunstack_tensorlistfromtensor]cell_while_tensorarrayv2read_tensorlistgetitem_cell_tensorarrayunstack_tensorlistfromtensor_0*6
_input_shapes%
#: : : : :	: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
: 

Χ
#__inference_cell_layer_call_fn_3810
inputs_0
gru_cell_variable
gru_cell_gru_cell_bias
gru_cell_gru_cell_kernel&
"gru_cell_gru_cell_recurrent_kernel
identity’StatefulPartitionedCallΐ
StatefulPartitionedCallStatefulPartitionedCallinputs_0gru_cell_variablegru_cell_gru_cell_biasgru_cell_gru_cell_kernel"gru_cell_gru_cell_recurrent_kernel*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_cell_layer_call_and_return_conditional_losses_13622
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:?????????ΐ::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:?????????ΐ
"
_user_specified_name
inputs/0
ϋZ

=__inference_gru_layer_call_and_return_conditional_losses_3241

inputs7
3cell_gru_cell_readvariableop_gru_cell_gru_cell_bias@
<cell_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel;
7cell_gru_cell_matmul_1_readvariableop_gru_cell_variableN
Jcell_gru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel
identity’cell/AssignVariableOp’
cell/while
cell/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
cell/transpose/perm
cell/transpose	Transposeinputscell/transpose/perm:output:0*
T0*#
_output_shapes
:+ΐ2
cell/transposem

cell/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"+      @  2

cell/Shape~
cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
cell/strided_slice/stack
cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cell/strided_slice/stack_1
cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cell/strided_slice/stack_2
cell/strided_sliceStridedSlicecell/Shape:output:0!cell/strided_slice/stack:output:0#cell/strided_slice/stack_1:output:0#cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cell/strided_slice
 cell/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 cell/TensorArrayV2/element_shapeΔ
cell/TensorArrayV2TensorListReserve)cell/TensorArrayV2/element_shape:output:0cell/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
cell/TensorArrayV2Ι
:cell/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   @  2<
:cell/TensorArrayUnstack/TensorListFromTensor/element_shape
,cell/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorcell/transpose:y:0Ccell/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02.
,cell/TensorArrayUnstack/TensorListFromTensor
cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
cell/strided_slice_1/stack
cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cell/strided_slice_1/stack_1
cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cell/strided_slice_1/stack_2
cell/strided_slice_1StridedSlicecell/transpose:y:0#cell/strided_slice_1/stack:output:0%cell/strided_slice_1/stack_1:output:0%cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ΐ*
shrink_axis_mask2
cell/strided_slice_1±
cell/gru_cell/ReadVariableOpReadVariableOp3cell_gru_cell_readvariableop_gru_cell_gru_cell_bias*
_output_shapes
:	*
dtype02
cell/gru_cell/ReadVariableOp
cell/gru_cell/unstackUnpack$cell/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
cell/gru_cell/unstackΙ
#cell/gru_cell/MatMul/ReadVariableOpReadVariableOp<cell_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel* 
_output_shapes
:
ΐ*
dtype02%
#cell/gru_cell/MatMul/ReadVariableOp¬
cell/gru_cell/MatMulMatMulcell/strided_slice_1:output:0+cell/gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
cell/gru_cell/MatMul£
cell/gru_cell/BiasAddBiasAddcell/gru_cell/MatMul:product:0cell/gru_cell/unstack:output:0*
T0*
_output_shapes
:	2
cell/gru_cell/BiasAddl
cell/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
cell/gru_cell/Const
cell/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
cell/gru_cell/split/split_dimΜ
cell/gru_cell/splitSplit&cell/gru_cell/split/split_dim:output:0cell/gru_cell/BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
cell/gru_cell/splitΗ
%cell/gru_cell/MatMul_1/ReadVariableOpReadVariableOp7cell_gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype02'
%cell/gru_cell/MatMul_1/ReadVariableOpί
'cell/gru_cell/MatMul_1/ReadVariableOp_1ReadVariableOpJcell_gru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel* 
_output_shapes
:
*
dtype02)
'cell/gru_cell/MatMul_1/ReadVariableOp_1Δ
cell/gru_cell/MatMul_1MatMul-cell/gru_cell/MatMul_1/ReadVariableOp:value:0/cell/gru_cell/MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
:	2
cell/gru_cell/MatMul_1©
cell/gru_cell/BiasAdd_1BiasAdd cell/gru_cell/MatMul_1:product:0cell/gru_cell/unstack:output:1*
T0*
_output_shapes
:	2
cell/gru_cell/BiasAdd_1
cell/gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ????2
cell/gru_cell/Const_1
cell/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
cell/gru_cell/split_1/split_dim
cell/gru_cell/split_1SplitV cell/gru_cell/BiasAdd_1:output:0cell/gru_cell/Const_1:output:0(cell/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2
cell/gru_cell/split_1
cell/gru_cell/addAddV2cell/gru_cell/split:output:0cell/gru_cell/split_1:output:0*
T0*
_output_shapes
:	2
cell/gru_cell/addz
cell/gru_cell/SigmoidSigmoidcell/gru_cell/add:z:0*
T0*
_output_shapes
:	2
cell/gru_cell/Sigmoid
cell/gru_cell/add_1AddV2cell/gru_cell/split:output:1cell/gru_cell/split_1:output:1*
T0*
_output_shapes
:	2
cell/gru_cell/add_1
cell/gru_cell/Sigmoid_1Sigmoidcell/gru_cell/add_1:z:0*
T0*
_output_shapes
:	2
cell/gru_cell/Sigmoid_1
cell/gru_cell/mulMulcell/gru_cell/Sigmoid_1:y:0cell/gru_cell/split_1:output:2*
T0*
_output_shapes
:	2
cell/gru_cell/mul
cell/gru_cell/add_2AddV2cell/gru_cell/split:output:2cell/gru_cell/mul:z:0*
T0*
_output_shapes
:	2
cell/gru_cell/add_2s
cell/gru_cell/TanhTanhcell/gru_cell/add_2:z:0*
T0*
_output_shapes
:	2
cell/gru_cell/TanhΑ
"cell/gru_cell/mul_1/ReadVariableOpReadVariableOp7cell_gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype02$
"cell/gru_cell/mul_1/ReadVariableOp’
cell/gru_cell/mul_1Mulcell/gru_cell/Sigmoid:y:0*cell/gru_cell/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
cell/gru_cell/mul_1o
cell/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
cell/gru_cell/sub/x
cell/gru_cell/subSubcell/gru_cell/sub/x:output:0cell/gru_cell/Sigmoid:y:0*
T0*
_output_shapes
:	2
cell/gru_cell/sub
cell/gru_cell/mul_2Mulcell/gru_cell/sub:z:0cell/gru_cell/Tanh:y:0*
T0*
_output_shapes
:	2
cell/gru_cell/mul_2
cell/gru_cell/add_3AddV2cell/gru_cell/mul_1:z:0cell/gru_cell/mul_2:z:0*
T0*
_output_shapes
:	2
cell/gru_cell/add_3
"cell/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2$
"cell/TensorArrayV2_1/element_shapeΚ
cell/TensorArrayV2_1TensorListReserve+cell/TensorArrayV2_1/element_shape:output:0cell/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
cell/TensorArrayV2_1X
	cell/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
	cell/time£
cell/ReadVariableOpReadVariableOp7cell_gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype02
cell/ReadVariableOp
cell/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
cell/while/maximum_iterationst
cell/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
cell/while/loop_counter

cell/whileWhile cell/while/loop_counter:output:0&cell/while/maximum_iterations:output:0cell/time:output:0cell/TensorArrayV2_1:handle:0cell/ReadVariableOp:value:0cell/strided_slice:output:0<cell/TensorArrayUnstack/TensorListFromTensor:output_handle:03cell_gru_cell_readvariableop_gru_cell_gru_cell_bias<cell_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernelJcell_gru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*1
_output_shapes
: : : : :	: : : : : *%
_read_only_resource_inputs
	* 
bodyR
cell_while_body_3149* 
condR
cell_while_cond_3148*0
output_shapes
: : : : :	: : : : : *
parallel_iterations 2

cell/whileΏ
5cell/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      27
5cell/TensorArrayV2Stack/TensorListStack/element_shapeτ
'cell/TensorArrayV2Stack/TensorListStackTensorListStackcell/while:output:3>cell/TensorArrayV2Stack/TensorListStack/element_shape:output:0*#
_output_shapes
:+*
element_dtype02)
'cell/TensorArrayV2Stack/TensorListStack
cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
cell/strided_slice_2/stack
cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
cell/strided_slice_2/stack_1
cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cell/strided_slice_2/stack_2°
cell/strided_slice_2StridedSlice0cell/TensorArrayV2Stack/TensorListStack:tensor:0#cell/strided_slice_2/stack:output:0%cell/strided_slice_2/stack_1:output:0%cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	*
shrink_axis_mask2
cell/strided_slice_2
cell/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
cell/transpose_1/perm±
cell/transpose_1	Transpose0cell/TensorArrayV2Stack/TensorListStack:tensor:0cell/transpose_1/perm:output:0*
T0*#
_output_shapes
:+2
cell/transpose_1p
cell/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
cell/runtime
cell/AssignVariableOpAssignVariableOp7cell_gru_cell_matmul_1_readvariableop_gru_cell_variablecell/while:output:4^cell/ReadVariableOp&^cell/gru_cell/MatMul_1/ReadVariableOp#^cell/gru_cell/mul_1/ReadVariableOp*
_output_shapes
 *
dtype02
cell/AssignVariableOpb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimscell/strided_slice_2:output:0ExpandDims/dim:output:0*
T0*#
_output_shapes
:2

ExpandDims
IdentityIdentityExpandDims:output:0^cell/AssignVariableOp^cell/while*
T0*#
_output_shapes
:2

Identity"
identityIdentity:output:0*2
_input_shapes!
:+ΐ::::2.
cell/AssignVariableOpcell/AssignVariableOp2

cell/while
cell/while:K G
#
_output_shapes
:+ΐ
 
_user_specified_nameinputs
ό
Τ
"__inference_gru_layer_call_fn_3411

inputs
gru_cell_gru_cell_bias
gru_cell_gru_cell_kernel
gru_cell_variable&
"gru_cell_gru_cell_recurrent_kernel
identity’StatefulPartitionedCallΑ
StatefulPartitionedCallStatefulPartitionedCallinputsgru_cell_gru_cell_biasgru_cell_gru_cell_kernelgru_cell_variable"gru_cell_gru_cell_recurrent_kernel*
Tin	
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *F
fAR?
=__inference_gru_layer_call_and_return_conditional_losses_18572
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:2

Identity"
identityIdentity:output:0*2
_input_shapes!
:+ΐ::::22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:+ΐ
 
_user_specified_nameinputs
½>

while_body_3561
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0:
6while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0C
?while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0O
Kwhile_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor8
4while_gru_cell_readvariableop_gru_cell_gru_cell_biasA
=while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernelM
Iwhile_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernelΓ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   @  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeΛ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes
:	ΐ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemΆ
while/gru_cell/ReadVariableOpReadVariableOp6while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0*
_output_shapes
:	*
dtype02
while/gru_cell/ReadVariableOp
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
while/gru_cell/unstackΞ
$while/gru_cell/MatMul/ReadVariableOpReadVariableOp?while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0* 
_output_shapes
:
ΐ*
dtype02&
$while/gru_cell/MatMul/ReadVariableOpΒ
while/gru_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0,while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
while/gru_cell/MatMul§
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0while/gru_cell/unstack:output:0*
T0*
_output_shapes
:	2
while/gru_cell/BiasAddn
while/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell/Const
while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
while/gru_cell/split/split_dimΠ
while/gru_cell/splitSplit'while/gru_cell/split/split_dim:output:0while/gru_cell/BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
while/gru_cell/splitή
&while/gru_cell/MatMul_1/ReadVariableOpReadVariableOpKwhile_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0* 
_output_shapes
:
*
dtype02(
&while/gru_cell/MatMul_1/ReadVariableOp«
while/gru_cell/MatMul_1MatMulwhile_placeholder_2.while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
while/gru_cell/MatMul_1­
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0while/gru_cell/unstack:output:1*
T0*
_output_shapes
:	2
while/gru_cell/BiasAdd_1
while/gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ????2
while/gru_cell/Const_1
 while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 while/gru_cell/split_1/split_dim
while/gru_cell/split_1SplitV!while/gru_cell/BiasAdd_1:output:0while/gru_cell/Const_1:output:0)while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2
while/gru_cell/split_1
while/gru_cell/addAddV2while/gru_cell/split:output:0while/gru_cell/split_1:output:0*
T0*
_output_shapes
:	2
while/gru_cell/add}
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*
_output_shapes
:	2
while/gru_cell/Sigmoid
while/gru_cell/add_1AddV2while/gru_cell/split:output:1while/gru_cell/split_1:output:1*
T0*
_output_shapes
:	2
while/gru_cell/add_1
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*
_output_shapes
:	2
while/gru_cell/Sigmoid_1
while/gru_cell/mulMulwhile/gru_cell/Sigmoid_1:y:0while/gru_cell/split_1:output:2*
T0*
_output_shapes
:	2
while/gru_cell/mul
while/gru_cell/add_2AddV2while/gru_cell/split:output:2while/gru_cell/mul:z:0*
T0*
_output_shapes
:	2
while/gru_cell/add_2v
while/gru_cell/TanhTanhwhile/gru_cell/add_2:z:0*
T0*
_output_shapes
:	2
while/gru_cell/Tanh
while/gru_cell/mul_1Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*
_output_shapes
:	2
while/gru_cell/mul_1q
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell/sub/x
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*
_output_shapes
:	2
while/gru_cell/sub
while/gru_cell/mul_2Mulwhile/gru_cell/sub:z:0while/gru_cell/Tanh:y:0*
T0*
_output_shapes
:	2
while/gru_cell/mul_2
while/gru_cell/add_3AddV2while/gru_cell/mul_1:z:0while/gru_cell/mul_2:z:0*
T0*
_output_shapes
:	2
while/gru_cell/add_3ά
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3t
while/Identity_4Identitywhile/gru_cell/add_3:z:0*
T0*
_output_shapes
:	2
while/Identity_4"
Iwhile_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernelKwhile_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0"
=while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel?while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0"n
4while_gru_cell_readvariableop_gru_cell_gru_cell_bias6while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*6
_input_shapes%
#: : : : :	: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
: 
μ
B
&__inference_dropout_layer_call_fn_3449

inputs
identity·
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_19132
PartitionedCalld
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs
Ψ

`
A__inference_dropout_layer_call_and_return_conditional_losses_1908

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *δ8?2
dropout/Constk
dropout/MulMulinputsdropout/Const:output:0*
T0*
_output_shapes
:	2
dropout/Mulo
dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2
dropout/Shape¬
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
_output_shapes
:	*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜΜ=2
dropout/GreaterEqual/yΆ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	2
dropout/GreaterEqualw
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	2
dropout/Castr
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*
_output_shapes
:	2
dropout/Mul_1]
IdentityIdentitydropout/Mul_1:z:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs


'__inference_stream_1_layer_call_fn_3071

inputs
stream_1_conv2d_1_kernel
stream_1_conv2d_1_bias
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstream_1_conv2d_1_kernelstream_1_conv2d_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:+$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_stream_1_layer_call_and_return_conditional_losses_15202
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:+$2

Identity"
identityIdentity:output:0*-
_input_shapes
:/&::22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:/&
 
_user_specified_nameinputs
³
³
A__inference_dense_2_layer_call_and_return_conditional_losses_3494

inputs(
$matmul_readvariableop_dense_2_kernel'
#biasadd_readvariableop_dense_2_bias
identity
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_2_kernel*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
MatMul
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_2_bias*
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
identityIdentity:output:0*&
_input_shapes
:	:::G C

_output_shapes
:	
 
_user_specified_nameinputs
­
»
$functional_1_gru_cell_while_cond_762H
Dfunctional_1_gru_cell_while_functional_1_gru_cell_while_loop_counterN
Jfunctional_1_gru_cell_while_functional_1_gru_cell_while_maximum_iterations+
'functional_1_gru_cell_while_placeholder-
)functional_1_gru_cell_while_placeholder_1-
)functional_1_gru_cell_while_placeholder_2H
Dfunctional_1_gru_cell_while_less_functional_1_gru_cell_strided_slice]
Yfunctional_1_gru_cell_while_functional_1_gru_cell_while_cond_762___redundant_placeholder0]
Yfunctional_1_gru_cell_while_functional_1_gru_cell_while_cond_762___redundant_placeholder1]
Yfunctional_1_gru_cell_while_functional_1_gru_cell_while_cond_762___redundant_placeholder2]
Yfunctional_1_gru_cell_while_functional_1_gru_cell_while_cond_762___redundant_placeholder3(
$functional_1_gru_cell_while_identity
ά
 functional_1/gru/cell/while/LessLess'functional_1_gru_cell_while_placeholderDfunctional_1_gru_cell_while_less_functional_1_gru_cell_strided_slice*
T0*
_output_shapes
: 2"
 functional_1/gru/cell/while/Less
$functional_1/gru/cell/while/IdentityIdentity$functional_1/gru/cell/while/Less:z:0*
T0
*
_output_shapes
: 2&
$functional_1/gru/cell/while/Identity"U
$functional_1_gru_cell_while_identity-functional_1/gru/cell/while/Identity:output:0*8
_input_shapes'
%: : : : :	: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
:
²

while_cond_1300
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice2
.while_while_cond_1300___redundant_placeholder02
.while_while_cond_1300___redundant_placeholder12
.while_while_cond_1300___redundant_placeholder22
.while_while_cond_1300___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*8
_input_shapes'
%: : : : :	: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
:
Ϊ]
ΰ
$functional_1_gru_cell_while_body_763H
Dfunctional_1_gru_cell_while_functional_1_gru_cell_while_loop_counterN
Jfunctional_1_gru_cell_while_functional_1_gru_cell_while_maximum_iterations+
'functional_1_gru_cell_while_placeholder-
)functional_1_gru_cell_while_placeholder_1-
)functional_1_gru_cell_while_placeholder_2E
Afunctional_1_gru_cell_while_functional_1_gru_cell_strided_slice_0
functional_1_gru_cell_while_tensorarrayv2read_tensorlistgetitem_functional_1_gru_cell_tensorarrayunstack_tensorlistfromtensor_0P
Lfunctional_1_gru_cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0Y
Ufunctional_1_gru_cell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0e
afunctional_1_gru_cell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0(
$functional_1_gru_cell_while_identity*
&functional_1_gru_cell_while_identity_1*
&functional_1_gru_cell_while_identity_2*
&functional_1_gru_cell_while_identity_3*
&functional_1_gru_cell_while_identity_4C
?functional_1_gru_cell_while_functional_1_gru_cell_strided_slice
}functional_1_gru_cell_while_tensorarrayv2read_tensorlistgetitem_functional_1_gru_cell_tensorarrayunstack_tensorlistfromtensorN
Jfunctional_1_gru_cell_while_gru_cell_readvariableop_gru_cell_gru_cell_biasW
Sfunctional_1_gru_cell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernelc
_functional_1_gru_cell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernelο
Mfunctional_1/gru/cell/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   @  2O
Mfunctional_1/gru/cell/while/TensorArrayV2Read/TensorListGetItem/element_shapeΟ
?functional_1/gru/cell/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemfunctional_1_gru_cell_while_tensorarrayv2read_tensorlistgetitem_functional_1_gru_cell_tensorarrayunstack_tensorlistfromtensor_0'functional_1_gru_cell_while_placeholderVfunctional_1/gru/cell/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes
:	ΐ*
element_dtype02A
?functional_1/gru/cell/while/TensorArrayV2Read/TensorListGetItemψ
3functional_1/gru/cell/while/gru_cell/ReadVariableOpReadVariableOpLfunctional_1_gru_cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0*
_output_shapes
:	*
dtype025
3functional_1/gru/cell/while/gru_cell/ReadVariableOpΫ
,functional_1/gru/cell/while/gru_cell/unstackUnpack;functional_1/gru/cell/while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2.
,functional_1/gru/cell/while/gru_cell/unstack
:functional_1/gru/cell/while/gru_cell/MatMul/ReadVariableOpReadVariableOpUfunctional_1_gru_cell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0* 
_output_shapes
:
ΐ*
dtype02<
:functional_1/gru/cell/while/gru_cell/MatMul/ReadVariableOp
+functional_1/gru/cell/while/gru_cell/MatMulMatMulFfunctional_1/gru/cell/while/TensorArrayV2Read/TensorListGetItem:item:0Bfunctional_1/gru/cell/while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2-
+functional_1/gru/cell/while/gru_cell/MatMul?
,functional_1/gru/cell/while/gru_cell/BiasAddBiasAdd5functional_1/gru/cell/while/gru_cell/MatMul:product:05functional_1/gru/cell/while/gru_cell/unstack:output:0*
T0*
_output_shapes
:	2.
,functional_1/gru/cell/while/gru_cell/BiasAdd
*functional_1/gru/cell/while/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2,
*functional_1/gru/cell/while/gru_cell/Const·
4functional_1/gru/cell/while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????26
4functional_1/gru/cell/while/gru_cell/split/split_dim¨
*functional_1/gru/cell/while/gru_cell/splitSplit=functional_1/gru/cell/while/gru_cell/split/split_dim:output:05functional_1/gru/cell/while/gru_cell/BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2,
*functional_1/gru/cell/while/gru_cell/split 
<functional_1/gru/cell/while/gru_cell/MatMul_1/ReadVariableOpReadVariableOpafunctional_1_gru_cell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0* 
_output_shapes
:
*
dtype02>
<functional_1/gru/cell/while/gru_cell/MatMul_1/ReadVariableOp
-functional_1/gru/cell/while/gru_cell/MatMul_1MatMul)functional_1_gru_cell_while_placeholder_2Dfunctional_1/gru/cell/while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2/
-functional_1/gru/cell/while/gru_cell/MatMul_1
.functional_1/gru/cell/while/gru_cell/BiasAdd_1BiasAdd7functional_1/gru/cell/while/gru_cell/MatMul_1:product:05functional_1/gru/cell/while/gru_cell/unstack:output:1*
T0*
_output_shapes
:	20
.functional_1/gru/cell/while/gru_cell/BiasAdd_1±
,functional_1/gru/cell/while/gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ????2.
,functional_1/gru/cell/while/gru_cell/Const_1»
6functional_1/gru/cell/while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????28
6functional_1/gru/cell/while/gru_cell/split_1/split_dimτ
,functional_1/gru/cell/while/gru_cell/split_1SplitV7functional_1/gru/cell/while/gru_cell/BiasAdd_1:output:05functional_1/gru/cell/while/gru_cell/Const_1:output:0?functional_1/gru/cell/while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2.
,functional_1/gru/cell/while/gru_cell/split_1σ
(functional_1/gru/cell/while/gru_cell/addAddV23functional_1/gru/cell/while/gru_cell/split:output:05functional_1/gru/cell/while/gru_cell/split_1:output:0*
T0*
_output_shapes
:	2*
(functional_1/gru/cell/while/gru_cell/addΏ
,functional_1/gru/cell/while/gru_cell/SigmoidSigmoid,functional_1/gru/cell/while/gru_cell/add:z:0*
T0*
_output_shapes
:	2.
,functional_1/gru/cell/while/gru_cell/Sigmoidχ
*functional_1/gru/cell/while/gru_cell/add_1AddV23functional_1/gru/cell/while/gru_cell/split:output:15functional_1/gru/cell/while/gru_cell/split_1:output:1*
T0*
_output_shapes
:	2,
*functional_1/gru/cell/while/gru_cell/add_1Ε
.functional_1/gru/cell/while/gru_cell/Sigmoid_1Sigmoid.functional_1/gru/cell/while/gru_cell/add_1:z:0*
T0*
_output_shapes
:	20
.functional_1/gru/cell/while/gru_cell/Sigmoid_1π
(functional_1/gru/cell/while/gru_cell/mulMul2functional_1/gru/cell/while/gru_cell/Sigmoid_1:y:05functional_1/gru/cell/while/gru_cell/split_1:output:2*
T0*
_output_shapes
:	2*
(functional_1/gru/cell/while/gru_cell/mulξ
*functional_1/gru/cell/while/gru_cell/add_2AddV23functional_1/gru/cell/while/gru_cell/split:output:2,functional_1/gru/cell/while/gru_cell/mul:z:0*
T0*
_output_shapes
:	2,
*functional_1/gru/cell/while/gru_cell/add_2Έ
)functional_1/gru/cell/while/gru_cell/TanhTanh.functional_1/gru/cell/while/gru_cell/add_2:z:0*
T0*
_output_shapes
:	2+
)functional_1/gru/cell/while/gru_cell/Tanhζ
*functional_1/gru/cell/while/gru_cell/mul_1Mul0functional_1/gru/cell/while/gru_cell/Sigmoid:y:0)functional_1_gru_cell_while_placeholder_2*
T0*
_output_shapes
:	2,
*functional_1/gru/cell/while/gru_cell/mul_1
*functional_1/gru/cell/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2,
*functional_1/gru/cell/while/gru_cell/sub/xμ
(functional_1/gru/cell/while/gru_cell/subSub3functional_1/gru/cell/while/gru_cell/sub/x:output:00functional_1/gru/cell/while/gru_cell/Sigmoid:y:0*
T0*
_output_shapes
:	2*
(functional_1/gru/cell/while/gru_cell/subζ
*functional_1/gru/cell/while/gru_cell/mul_2Mul,functional_1/gru/cell/while/gru_cell/sub:z:0-functional_1/gru/cell/while/gru_cell/Tanh:y:0*
T0*
_output_shapes
:	2,
*functional_1/gru/cell/while/gru_cell/mul_2λ
*functional_1/gru/cell/while/gru_cell/add_3AddV2.functional_1/gru/cell/while/gru_cell/mul_1:z:0.functional_1/gru/cell/while/gru_cell/mul_2:z:0*
T0*
_output_shapes
:	2,
*functional_1/gru/cell/while/gru_cell/add_3Κ
@functional_1/gru/cell/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)functional_1_gru_cell_while_placeholder_1'functional_1_gru_cell_while_placeholder.functional_1/gru/cell/while/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype02B
@functional_1/gru/cell/while/TensorArrayV2Write/TensorListSetItem
!functional_1/gru/cell/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!functional_1/gru/cell/while/add/yΑ
functional_1/gru/cell/while/addAddV2'functional_1_gru_cell_while_placeholder*functional_1/gru/cell/while/add/y:output:0*
T0*
_output_shapes
: 2!
functional_1/gru/cell/while/add
#functional_1/gru/cell/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#functional_1/gru/cell/while/add_1/yδ
!functional_1/gru/cell/while/add_1AddV2Dfunctional_1_gru_cell_while_functional_1_gru_cell_while_loop_counter,functional_1/gru/cell/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!functional_1/gru/cell/while/add_1 
$functional_1/gru/cell/while/IdentityIdentity%functional_1/gru/cell/while/add_1:z:0*
T0*
_output_shapes
: 2&
$functional_1/gru/cell/while/IdentityΙ
&functional_1/gru/cell/while/Identity_1IdentityJfunctional_1_gru_cell_while_functional_1_gru_cell_while_maximum_iterations*
T0*
_output_shapes
: 2(
&functional_1/gru/cell/while/Identity_1’
&functional_1/gru/cell/while/Identity_2Identity#functional_1/gru/cell/while/add:z:0*
T0*
_output_shapes
: 2(
&functional_1/gru/cell/while/Identity_2Ο
&functional_1/gru/cell/while/Identity_3IdentityPfunctional_1/gru/cell/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2(
&functional_1/gru/cell/while/Identity_3Ά
&functional_1/gru/cell/while/Identity_4Identity.functional_1/gru/cell/while/gru_cell/add_3:z:0*
T0*
_output_shapes
:	2(
&functional_1/gru/cell/while/Identity_4"
?functional_1_gru_cell_while_functional_1_gru_cell_strided_sliceAfunctional_1_gru_cell_while_functional_1_gru_cell_strided_slice_0"Δ
_functional_1_gru_cell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernelafunctional_1_gru_cell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0"¬
Sfunctional_1_gru_cell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernelUfunctional_1_gru_cell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0"
Jfunctional_1_gru_cell_while_gru_cell_readvariableop_gru_cell_gru_cell_biasLfunctional_1_gru_cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0"U
$functional_1_gru_cell_while_identity-functional_1/gru/cell/while/Identity:output:0"Y
&functional_1_gru_cell_while_identity_1/functional_1/gru/cell/while/Identity_1:output:0"Y
&functional_1_gru_cell_while_identity_2/functional_1/gru/cell/while/Identity_2:output:0"Y
&functional_1_gru_cell_while_identity_3/functional_1/gru/cell/while/Identity_3:output:0"Y
&functional_1_gru_cell_while_identity_4/functional_1/gru/cell/while/Identity_4:output:0"
}functional_1_gru_cell_while_tensorarrayv2read_tensorlistgetitem_functional_1_gru_cell_tensorarrayunstack_tensorlistfromtensorfunctional_1_gru_cell_while_tensorarrayv2read_tensorlistgetitem_functional_1_gru_cell_tensorarrayunstack_tensorlistfromtensor_0*6
_input_shapes%
#: : : : :	: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
: 

B
&__inference_reshape_layer_call_fn_3089

inputs
identity»
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:+ΐ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_15452
PartitionedCallh
IdentityIdentityPartitionedCall:output:0*
T0*#
_output_shapes
:+ΐ2

Identity"
identityIdentity:output:0*%
_input_shapes
:+$:N J
&
_output_shapes
:+$
 
_user_specified_nameinputs

Q
5__inference_tf_op_layer_ExpandDims_layer_call_fn_3035

inputs
identityΝ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:1(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_14782
PartitionedCallk
IdentityIdentityPartitionedCall:output:0*
T0*&
_output_shapes
:1(2

Identity"
identityIdentity:output:0*!
_input_shapes
:1(:J F
"
_output_shapes
:1(
 
_user_specified_nameinputs
Δ
ύ
cell_while_cond_1764&
"cell_while_cell_while_loop_counter,
(cell_while_cell_while_maximum_iterations
cell_while_placeholder
cell_while_placeholder_1
cell_while_placeholder_2&
"cell_while_less_cell_strided_slice<
8cell_while_cell_while_cond_1764___redundant_placeholder0<
8cell_while_cell_while_cond_1764___redundant_placeholder1<
8cell_while_cell_while_cond_1764___redundant_placeholder2<
8cell_while_cell_while_cond_1764___redundant_placeholder3
cell_while_identity

cell/while/LessLesscell_while_placeholder"cell_while_less_cell_strided_slice*
T0*
_output_shapes
: 2
cell/while/Lessl
cell/while/IdentityIdentitycell/while/Less:z:0*
T0
*
_output_shapes
: 2
cell/while/Identity"3
cell_while_identitycell/while/Identity:output:0*8
_input_shapes'
%: : : : :	: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
:
ψ
_
&__inference_dropout_layer_call_fn_3444

inputs
identity’StatefulPartitionedCallΟ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_19082
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*
_input_shapes
:	22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	
 
_user_specified_nameinputs
?

B__inference_gru_cell_layer_call_and_return_conditional_losses_3951

inputs

states)
%readvariableop_gru_cell_gru_cell_bias2
.matmul_readvariableop_gru_cell_gru_cell_kernel@
<matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel
identity

identity_1
ReadVariableOpReadVariableOp%readvariableop_gru_cell_gru_cell_bias*
_output_shapes
:	*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOp.matmul_readvariableop_gru_cell_gru_cell_kernel* 
_output_shapes
:
ΐ*
dtype02
MatMul/ReadVariableOpk
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
MatMulk
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*
_output_shapes
:	2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split/split_dim
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
splits
MatMul_1/ReadVariableOpReadVariableOpstates*
_output_shapes
:*
dtype02
MatMul_1/ReadVariableOp΅
MatMul_1/ReadVariableOp_1ReadVariableOp<matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOp_1
MatMul_1BatchMatMulV2MatMul_1/ReadVariableOp:value:0!MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
:2

MatMul_1i
	BiasAdd_1BiasAddMatMul_1:output:0unstack:output:1*
T0*
_output_shapes
:2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ????2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split_1/split_dim¦
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0* 
_output_shapes
:::*
	num_split2	
split_1X
addAddV2split:output:0split_1:output:0*
T0*
_output_shapes
:2
addI
SigmoidSigmoidadd:z:0*
T0*
_output_shapes
:2	
Sigmoid\
add_1AddV2split:output:1split_1:output:1*
T0*
_output_shapes
:2
add_1O
	Sigmoid_1Sigmoid	add_1:z:0*
T0*
_output_shapes
:2
	Sigmoid_1U
mulMulSigmoid_1:y:0split_1:output:2*
T0*
_output_shapes
:2
mulS
add_2AddV2split:output:2mul:z:0*
T0*
_output_shapes
:2
add_2B
TanhTanh	add_2:z:0*
T0*
_output_shapes
:2
Tanhm
mul_1/ReadVariableOpReadVariableOpstates*
_output_shapes
:*
dtype02
mul_1/ReadVariableOpc
mul_1MulSigmoid:y:0mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xQ
subSubsub/x:output:0Sigmoid:y:0*
T0*
_output_shapes
:2
subK
mul_2Mulsub:z:0Tanh:y:0*
T0*
_output_shapes
:2
mul_2P
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*
_output_shapes
:2
add_3N
IdentityIdentity	add_3:z:0*
T0*
_output_shapes
:2

IdentityR

Identity_1Identity	add_3:z:0*
T0*
_output_shapes
:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*.
_input_shapes
:	ΐ:::::G C

_output_shapes
:	ΐ
 
_user_specified_nameinputs:&"
 
_user_specified_namestates
½>

while_body_3711
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0:
6while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0C
?while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0O
Kwhile_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor8
4while_gru_cell_readvariableop_gru_cell_gru_cell_biasA
=while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernelM
Iwhile_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernelΓ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   @  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeΛ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes
:	ΐ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemΆ
while/gru_cell/ReadVariableOpReadVariableOp6while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0*
_output_shapes
:	*
dtype02
while/gru_cell/ReadVariableOp
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
while/gru_cell/unstackΞ
$while/gru_cell/MatMul/ReadVariableOpReadVariableOp?while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0* 
_output_shapes
:
ΐ*
dtype02&
$while/gru_cell/MatMul/ReadVariableOpΒ
while/gru_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0,while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
while/gru_cell/MatMul§
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0while/gru_cell/unstack:output:0*
T0*
_output_shapes
:	2
while/gru_cell/BiasAddn
while/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell/Const
while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
while/gru_cell/split/split_dimΠ
while/gru_cell/splitSplit'while/gru_cell/split/split_dim:output:0while/gru_cell/BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
while/gru_cell/splitή
&while/gru_cell/MatMul_1/ReadVariableOpReadVariableOpKwhile_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0* 
_output_shapes
:
*
dtype02(
&while/gru_cell/MatMul_1/ReadVariableOp«
while/gru_cell/MatMul_1MatMulwhile_placeholder_2.while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
while/gru_cell/MatMul_1­
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0while/gru_cell/unstack:output:1*
T0*
_output_shapes
:	2
while/gru_cell/BiasAdd_1
while/gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ????2
while/gru_cell/Const_1
 while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 while/gru_cell/split_1/split_dim
while/gru_cell/split_1SplitV!while/gru_cell/BiasAdd_1:output:0while/gru_cell/Const_1:output:0)while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2
while/gru_cell/split_1
while/gru_cell/addAddV2while/gru_cell/split:output:0while/gru_cell/split_1:output:0*
T0*
_output_shapes
:	2
while/gru_cell/add}
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*
_output_shapes
:	2
while/gru_cell/Sigmoid
while/gru_cell/add_1AddV2while/gru_cell/split:output:1while/gru_cell/split_1:output:1*
T0*
_output_shapes
:	2
while/gru_cell/add_1
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*
_output_shapes
:	2
while/gru_cell/Sigmoid_1
while/gru_cell/mulMulwhile/gru_cell/Sigmoid_1:y:0while/gru_cell/split_1:output:2*
T0*
_output_shapes
:	2
while/gru_cell/mul
while/gru_cell/add_2AddV2while/gru_cell/split:output:2while/gru_cell/mul:z:0*
T0*
_output_shapes
:	2
while/gru_cell/add_2v
while/gru_cell/TanhTanhwhile/gru_cell/add_2:z:0*
T0*
_output_shapes
:	2
while/gru_cell/Tanh
while/gru_cell/mul_1Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*
_output_shapes
:	2
while/gru_cell/mul_1q
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell/sub/x
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*
_output_shapes
:	2
while/gru_cell/sub
while/gru_cell/mul_2Mulwhile/gru_cell/sub:z:0while/gru_cell/Tanh:y:0*
T0*
_output_shapes
:	2
while/gru_cell/mul_2
while/gru_cell/add_3AddV2while/gru_cell/mul_1:z:0while/gru_cell/mul_2:z:0*
T0*
_output_shapes
:	2
while/gru_cell/add_3ά
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3t
while/Identity_4Identitywhile/gru_cell/add_3:z:0*
T0*
_output_shapes
:	2
while/Identity_4"
Iwhile_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernelKwhile_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0"
=while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel?while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0"n
4while_gru_cell_readvariableop_gru_cell_gru_cell_bias6while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*6
_input_shapes%
#: : : : :	: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
: 
ύ

%__inference_stream_layer_call_fn_3053

inputs
stream_conv2d_kernel
stream_conv2d_bias
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstream_conv2d_kernelstream_conv2d_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:/&*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_stream_layer_call_and_return_conditional_losses_14972
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:/&2

Identity"
identityIdentity:output:0*-
_input_shapes
:1(::22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:1(
 
_user_specified_nameinputs


A__inference_gru_cell_layer_call_and_return_conditional_losses_939

inputs

states)
%readvariableop_gru_cell_gru_cell_bias2
.matmul_readvariableop_gru_cell_gru_cell_kernel@
<matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel
identity

identity_1
ReadVariableOpReadVariableOp%readvariableop_gru_cell_gru_cell_bias*
_output_shapes
:	*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOp.matmul_readvariableop_gru_cell_gru_cell_kernel* 
_output_shapes
:
ΐ*
dtype02
MatMul/ReadVariableOpk
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
MatMulk
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*
_output_shapes
:	2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split/split_dim
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
splitz
MatMul_1/ReadVariableOpReadVariableOpstates*
_output_shapes
:	*
dtype02
MatMul_1/ReadVariableOp΅
MatMul_1/ReadVariableOp_1ReadVariableOp<matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOp_1
MatMul_1MatMulMatMul_1/ReadVariableOp:value:0!MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
:	2

MatMul_1q
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*
_output_shapes
:	2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ????2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split_1/split_dim»
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2	
split_1_
addAddV2split:output:0split_1:output:0*
T0*
_output_shapes
:	2
addP
SigmoidSigmoidadd:z:0*
T0*
_output_shapes
:	2	
Sigmoidc
add_1AddV2split:output:1split_1:output:1*
T0*
_output_shapes
:	2
add_1V
	Sigmoid_1Sigmoid	add_1:z:0*
T0*
_output_shapes
:	2
	Sigmoid_1\
mulMulSigmoid_1:y:0split_1:output:2*
T0*
_output_shapes
:	2
mulZ
add_2AddV2split:output:2mul:z:0*
T0*
_output_shapes
:	2
add_2I
TanhTanh	add_2:z:0*
T0*
_output_shapes
:	2
Tanht
mul_1/ReadVariableOpReadVariableOpstates*
_output_shapes
:	*
dtype02
mul_1/ReadVariableOpj
mul_1MulSigmoid:y:0mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xX
subSubsub/x:output:0Sigmoid:y:0*
T0*
_output_shapes
:	2
subR
mul_2Mulsub:z:0Tanh:y:0*
T0*
_output_shapes
:	2
mul_2W
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*
_output_shapes
:	2
add_3U
IdentityIdentity	add_3:z:0*
T0*
_output_shapes
:	2

IdentityY

Identity_1Identity	add_3:z:0*
T0*
_output_shapes
:	2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*.
_input_shapes
:	ΐ:::::G C

_output_shapes
:	ΐ
 
_user_specified_nameinputs:&"
 
_user_specified_namestates
ΐ
^
B__inference_stream_2_layer_call_and_return_conditional_losses_3417

inputs
identityo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten/Constw
flatten/ReshapeReshapeinputsflatten/Const:output:0*
T0*
_output_shapes
:	2
flatten/Reshaped
IdentityIdentityflatten/Reshape:output:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*"
_input_shapes
::K G
#
_output_shapes
:
 
_user_specified_nameinputs
Έ	
Ι
gru_cell_while_cond_2222.
*gru_cell_while_gru_cell_while_loop_counter4
0gru_cell_while_gru_cell_while_maximum_iterations
gru_cell_while_placeholder 
gru_cell_while_placeholder_1 
gru_cell_while_placeholder_2.
*gru_cell_while_less_gru_cell_strided_sliceD
@gru_cell_while_gru_cell_while_cond_2222___redundant_placeholder0D
@gru_cell_while_gru_cell_while_cond_2222___redundant_placeholder1D
@gru_cell_while_gru_cell_while_cond_2222___redundant_placeholder2D
@gru_cell_while_gru_cell_while_cond_2222___redundant_placeholder3
gru_cell_while_identity

gru/cell/while/LessLessgru_cell_while_placeholder*gru_cell_while_less_gru_cell_strided_slice*
T0*
_output_shapes
: 2
gru/cell/while/Lessx
gru/cell/while/IdentityIdentitygru/cell/while/Less:z:0*
T0
*
_output_shapes
: 2
gru/cell/while/Identity";
gru_cell_while_identity gru/cell/while/Identity:output:0*8
_input_shapes'
%: : : : :	: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
:
³
³
A__inference_dense_2_layer_call_and_return_conditional_losses_1981

inputs(
$matmul_readvariableop_dense_2_kernel'
#biasadd_readvariableop_dense_2_bias
identity
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_2_kernel*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
MatMul
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_2_bias*
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
identityIdentity:output:0*&
_input_shapes
:	:::G C

_output_shapes
:	
 
_user_specified_nameinputs
Ά4
Έ
>__inference_cell_layer_call_and_return_conditional_losses_1362

inputs
gru_cell_gru_cell_variable#
gru_cell_gru_cell_gru_cell_bias%
!gru_cell_gru_cell_gru_cell_kernel/
+gru_cell_gru_cell_gru_cell_recurrent_kernel
identity’AssignVariableOp’ gru_cell/StatefulPartitionedCall’whileu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:?????????ΐ2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2β
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape°
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2Ώ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   @  27
5TensorArrayUnstack/TensorListFromTensor/element_shapeψ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2τ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ΐ*
shrink_axis_mask2
strided_slice_1
 gru_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_1:output:0gru_cell_gru_cell_variablegru_cell_gru_cell_gru_cell_bias!gru_cell_gru_cell_gru_cell_kernel+gru_cell_gru_cell_gru_cell_recurrent_kernel*
Tin	
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:	:	*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_gru_cell_layer_call_and_return_conditional_losses_9392"
 gru_cell/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
TensorArrayV2_1/element_shapeΆ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time|
ReadVariableOpReadVariableOpgru_cell_gru_cell_variable*
_output_shapes
:	*
dtype02
ReadVariableOp
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0ReadVariableOp:value:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_gru_cell_gru_cell_bias!gru_cell_gru_cell_gru_cell_kernel+gru_cell_gru_cell_gru_cell_recurrent_kernel*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*1
_output_shapes
: : : : :	: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_1301*
condR
while_cond_1300*0
output_shapes
: : : : :	: : : : : *
parallel_iterations 2
while΅
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      22
0TensorArrayV2Stack/TensorListStack/element_shapeι
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:?????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¦
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:?????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime»
AssignVariableOpAssignVariableOpgru_cell_gru_cell_variablewhile:output:4^ReadVariableOp!^gru_cell/StatefulPartitionedCall*
_output_shapes
 *
dtype02
AssignVariableOp’
IdentityIdentitystrided_slice_2:output:0^AssignVariableOp!^gru_cell/StatefulPartitionedCall^while*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:?????????ΐ::::2$
AssignVariableOpAssignVariableOp2D
 gru_cell/StatefulPartitionedCall gru_cell/StatefulPartitionedCall2
whilewhile:T P
,
_output_shapes
:?????????ΐ
 
_user_specified_nameinputs
#
Ό
while_body_1400
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0+
'while_gru_cell_gru_cell_gru_cell_bias_0-
)while_gru_cell_gru_cell_gru_cell_kernel_07
3while_gru_cell_gru_cell_gru_cell_recurrent_kernel_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor)
%while_gru_cell_gru_cell_gru_cell_bias+
'while_gru_cell_gru_cell_gru_cell_kernel5
1while_gru_cell_gru_cell_gru_cell_recurrent_kernel’&while/gru_cell/StatefulPartitionedCallΓ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   @  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeΛ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes
:	ΐ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemΛ
&while/gru_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2'while_gru_cell_gru_cell_gru_cell_bias_0)while_gru_cell_gru_cell_gru_cell_kernel_03while_gru_cell_gru_cell_gru_cell_recurrent_kernel_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:	:	*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_gru_cell_layer_call_and_return_conditional_losses_11172(
&while/gru_cell/StatefulPartitionedCallσ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder/while/gru_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:0'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2Ά
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3΄
while/Identity_4Identity/while/gru_cell/StatefulPartitionedCall:output:1'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
:	2
while/Identity_4"P
%while_gru_cell_gru_cell_gru_cell_bias'while_gru_cell_gru_cell_gru_cell_bias_0"T
'while_gru_cell_gru_cell_gru_cell_kernel)while_gru_cell_gru_cell_gru_cell_kernel_0"h
1while_gru_cell_gru_cell_gru_cell_recurrent_kernel3while_gru_cell_gru_cell_gru_cell_recurrent_kernel_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*6
_input_shapes%
#: : : : :	: : :::2P
&while/gru_cell/StatefulPartitionedCall&while/gru_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
: 

―
"__inference_signature_wrapper_2138
input_1
stream_conv2d_kernel
stream_conv2d_bias
stream_1_conv2d_1_kernel
stream_1_conv2d_1_bias
gru_cell_gru_cell_bias
gru_cell_gru_cell_kernel
gru_cell_variable&
"gru_cell_gru_cell_recurrent_kernel
dense_kernel

dense_bias
dense_1_kernel
dense_1_bias
dense_2_kernel
dense_2_bias
identity’StatefulPartitionedCallδ
StatefulPartitionedCallStatefulPartitionedCallinput_1stream_conv2d_kernelstream_conv2d_biasstream_1_conv2d_1_kernelstream_1_conv2d_1_biasgru_cell_gru_cell_biasgru_cell_gru_cell_kernelgru_cell_variable"gru_cell_gru_cell_recurrent_kerneldense_kernel
dense_biasdense_1_kerneldense_1_biasdense_2_kerneldense_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*/
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__wrapped_model_8772
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:1(::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:K G
"
_output_shapes
:1(
!
_user_specified_name	input_1
?
­
?__inference_dense_layer_call_and_return_conditional_losses_1936

inputs&
"matmul_readvariableop_dense_kernel%
!biasadd_readvariableop_dense_bias
identity
MatMul/ReadVariableOpReadVariableOp"matmul_readvariableop_dense_kernel* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpk
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
MatMul
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_dense_bias*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2	
BiasAdd\
IdentityIdentityBiasAdd:output:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*&
_input_shapes
:	:::G C

_output_shapes
:	
 
_user_specified_nameinputs

³
A__inference_dense_1_layer_call_and_return_conditional_losses_1959

inputs(
$matmul_readvariableop_dense_1_kernel'
#biasadd_readvariableop_dense_1_bias
identity
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_1_kernel* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpk
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
MatMul
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_1_bias*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2	
BiasAddP
ReluReluBiasAdd:output:0*
T0*
_output_shapes
:	2
Relu^
IdentityIdentityRelu:activations:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*&
_input_shapes
:	:::G C

_output_shapes
:	
 
_user_specified_nameinputs
ά-
Ϊ
F__inference_functional_1_layer_call_and_return_conditional_losses_2100

inputs
stream_stream_conv2d_kernel
stream_stream_conv2d_bias%
!stream_1_stream_1_conv2d_1_kernel#
stream_1_stream_1_conv2d_1_bias
gru_gru_cell_gru_cell_bias 
gru_gru_cell_gru_cell_kernel
gru_gru_cell_variable*
&gru_gru_cell_gru_cell_recurrent_kernel
dense_dense_kernel
dense_dense_bias
dense_1_dense_1_kernel
dense_1_dense_1_bias
dense_2_dense_2_kernel
dense_2_dense_2_bias
identity’dense/StatefulPartitionedCall’dense_1/StatefulPartitionedCall’dense_2/StatefulPartitionedCall’gru/StatefulPartitionedCall’stream/StatefulPartitionedCall’ stream_1/StatefulPartitionedCallϋ
&tf_op_layer_ExpandDims/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:1(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_14782(
&tf_op_layer_ExpandDims/PartitionedCallΚ
stream/StatefulPartitionedCallStatefulPartitionedCall/tf_op_layer_ExpandDims/PartitionedCall:output:0stream_stream_conv2d_kernelstream_stream_conv2d_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:/&*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_stream_layer_call_and_return_conditional_losses_14972 
stream/StatefulPartitionedCallΤ
 stream_1/StatefulPartitionedCallStatefulPartitionedCall'stream/StatefulPartitionedCall:output:0!stream_1_stream_1_conv2d_1_kernelstream_1_stream_1_conv2d_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:+$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_stream_1_layer_call_and_return_conditional_losses_15202"
 stream_1/StatefulPartitionedCallξ
reshape/PartitionedCallPartitionedCall)stream_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:+ΐ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_15452
reshape/PartitionedCallσ
gru/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0gru_gru_cell_gru_cell_biasgru_gru_cell_gru_cell_kernelgru_gru_cell_variable&gru_gru_cell_gru_cell_recurrent_kernel*
Tin	
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *F
fAR?
=__inference_gru_layer_call_and_return_conditional_losses_18572
gru/StatefulPartitionedCallθ
stream_2/PartitionedCallPartitionedCall$gru/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_stream_2_layer_call_and_return_conditional_losses_18882
stream_2/PartitionedCallβ
dropout/PartitionedCallPartitionedCall!stream_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_19132
dropout/PartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_dense_kerneldense_dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_19362
dense/StatefulPartitionedCall³
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_dense_1_kerneldense_1_dense_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_19592!
dense_1/StatefulPartitionedCall΄
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_dense_2_kerneldense_2_dense_2_bias*
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
GPU 2J 8 *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_19812!
dense_2/StatefulPartitionedCallΉ
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^gru/StatefulPartitionedCall^stream/StatefulPartitionedCall!^stream_1/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:1(::::::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2@
stream/StatefulPartitionedCallstream/StatefulPartitionedCall2D
 stream_1/StatefulPartitionedCall stream_1/StatefulPartitionedCall:S O
+
_output_shapes
:?????????1(
 
_user_specified_nameinputs
€
Ρ
F__inference_functional_1_layer_call_and_return_conditional_losses_2543

inputs<
8stream_conv2d_conv2d_readvariableop_stream_conv2d_kernel;
7stream_conv2d_biasadd_readvariableop_stream_conv2d_biasD
@stream_1_conv2d_1_conv2d_readvariableop_stream_1_conv2d_1_kernelC
?stream_1_conv2d_1_biasadd_readvariableop_stream_1_conv2d_1_bias;
7gru_cell_gru_cell_readvariableop_gru_cell_gru_cell_biasD
@gru_cell_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel?
;gru_cell_gru_cell_matmul_1_readvariableop_gru_cell_variableR
Ngru_cell_gru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel,
(dense_matmul_readvariableop_dense_kernel+
'dense_biasadd_readvariableop_dense_bias0
,dense_1_matmul_readvariableop_dense_1_kernel/
+dense_1_biasadd_readvariableop_dense_1_bias0
,dense_2_matmul_readvariableop_dense_2_kernel/
+dense_2_biasadd_readvariableop_dense_2_bias
identity’gru/cell/AssignVariableOp’gru/cell/while
%tf_op_layer_ExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%tf_op_layer_ExpandDims/ExpandDims/dimΜ
!tf_op_layer_ExpandDims/ExpandDims
ExpandDimsinputs.tf_op_layer_ExpandDims/ExpandDims/dim:output:0*
T0*
_cloned(*&
_output_shapes
:1(2#
!tf_op_layer_ExpandDims/ExpandDimsΛ
#stream/conv2d/Conv2D/ReadVariableOpReadVariableOp8stream_conv2d_conv2d_readvariableop_stream_conv2d_kernel*&
_output_shapes
:*
dtype02%
#stream/conv2d/Conv2D/ReadVariableOpι
stream/conv2d/Conv2DConv2D*tf_op_layer_ExpandDims/ExpandDims:output:0+stream/conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:/&*
paddingVALID*
strides
2
stream/conv2d/Conv2Dΐ
$stream/conv2d/BiasAdd/ReadVariableOpReadVariableOp7stream_conv2d_biasadd_readvariableop_stream_conv2d_bias*
_output_shapes
:*
dtype02&
$stream/conv2d/BiasAdd/ReadVariableOp·
stream/conv2d/BiasAddBiasAddstream/conv2d/Conv2D:output:0,stream/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:/&2
stream/conv2d/BiasAdd
stream/conv2d/ReluRelustream/conv2d/BiasAdd:output:0*
T0*&
_output_shapes
:/&2
stream/conv2d/ReluΫ
'stream_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp@stream_1_conv2d_1_conv2d_readvariableop_stream_1_conv2d_1_kernel*&
_output_shapes
:*
dtype02)
'stream_1/conv2d_1/Conv2D/ReadVariableOpλ
stream_1/conv2d_1/Conv2DConv2D stream/conv2d/Relu:activations:0/stream_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:+$*
paddingVALID*
strides
2
stream_1/conv2d_1/Conv2DΠ
(stream_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp?stream_1_conv2d_1_biasadd_readvariableop_stream_1_conv2d_1_bias*
_output_shapes
:*
dtype02*
(stream_1/conv2d_1/BiasAdd/ReadVariableOpΗ
stream_1/conv2d_1/BiasAddBiasAdd!stream_1/conv2d_1/Conv2D:output:00stream_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:+$2
stream_1/conv2d_1/BiasAdd
stream_1/conv2d_1/ReluRelu"stream_1/conv2d_1/BiasAdd:output:0*
T0*&
_output_shapes
:+$2
stream_1/conv2d_1/Reluw
reshape/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"   +   $      2
reshape/Shape
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slice}
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
reshape/Reshape/shape/1u
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :ΐ2
reshape/Reshape/shape/2Θ
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape‘
reshape/ReshapeReshape$stream_1/conv2d_1/Relu:activations:0reshape/Reshape/shape:output:0*
T0*#
_output_shapes
:+ΐ2
reshape/Reshape
gru/cell/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/cell/transpose/perm
gru/cell/transpose	Transposereshape/Reshape:output:0 gru/cell/transpose/perm:output:0*
T0*#
_output_shapes
:+ΐ2
gru/cell/transposeu
gru/cell/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"+      @  2
gru/cell/Shape
gru/cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/cell/strided_slice/stack
gru/cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru/cell/strided_slice/stack_1
gru/cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru/cell/strided_slice/stack_2
gru/cell/strided_sliceStridedSlicegru/cell/Shape:output:0%gru/cell/strided_slice/stack:output:0'gru/cell/strided_slice/stack_1:output:0'gru/cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru/cell/strided_slice
$gru/cell/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$gru/cell/TensorArrayV2/element_shapeΤ
gru/cell/TensorArrayV2TensorListReserve-gru/cell/TensorArrayV2/element_shape:output:0gru/cell/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru/cell/TensorArrayV2Ρ
>gru/cell/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   @  2@
>gru/cell/TensorArrayUnstack/TensorListFromTensor/element_shape
0gru/cell/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru/cell/transpose:y:0Ggru/cell/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type022
0gru/cell/TensorArrayUnstack/TensorListFromTensor
gru/cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru/cell/strided_slice_1/stack
 gru/cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru/cell/strided_slice_1/stack_1
 gru/cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru/cell/strided_slice_1/stack_2ͺ
gru/cell/strided_slice_1StridedSlicegru/cell/transpose:y:0'gru/cell/strided_slice_1/stack:output:0)gru/cell/strided_slice_1/stack_1:output:0)gru/cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ΐ*
shrink_axis_mask2
gru/cell/strided_slice_1½
 gru/cell/gru_cell/ReadVariableOpReadVariableOp7gru_cell_gru_cell_readvariableop_gru_cell_gru_cell_bias*
_output_shapes
:	*
dtype02"
 gru/cell/gru_cell/ReadVariableOp’
gru/cell/gru_cell/unstackUnpack(gru/cell/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
gru/cell/gru_cell/unstackΥ
'gru/cell/gru_cell/MatMul/ReadVariableOpReadVariableOp@gru_cell_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel* 
_output_shapes
:
ΐ*
dtype02)
'gru/cell/gru_cell/MatMul/ReadVariableOpΌ
gru/cell/gru_cell/MatMulMatMul!gru/cell/strided_slice_1:output:0/gru/cell/gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/MatMul³
gru/cell/gru_cell/BiasAddBiasAdd"gru/cell/gru_cell/MatMul:product:0"gru/cell/gru_cell/unstack:output:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/BiasAddt
gru/cell/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru/cell/gru_cell/Const
!gru/cell/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!gru/cell/gru_cell/split/split_dimά
gru/cell/gru_cell/splitSplit*gru/cell/gru_cell/split/split_dim:output:0"gru/cell/gru_cell/BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
gru/cell/gru_cell/splitΣ
)gru/cell/gru_cell/MatMul_1/ReadVariableOpReadVariableOp;gru_cell_gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype02+
)gru/cell/gru_cell/MatMul_1/ReadVariableOpλ
+gru/cell/gru_cell/MatMul_1/ReadVariableOp_1ReadVariableOpNgru_cell_gru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel* 
_output_shapes
:
*
dtype02-
+gru/cell/gru_cell/MatMul_1/ReadVariableOp_1Τ
gru/cell/gru_cell/MatMul_1MatMul1gru/cell/gru_cell/MatMul_1/ReadVariableOp:value:03gru/cell/gru_cell/MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/MatMul_1Ή
gru/cell/gru_cell/BiasAdd_1BiasAdd$gru/cell/gru_cell/MatMul_1:product:0"gru/cell/gru_cell/unstack:output:1*
T0*
_output_shapes
:	2
gru/cell/gru_cell/BiasAdd_1
gru/cell/gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ????2
gru/cell/gru_cell/Const_1
#gru/cell/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#gru/cell/gru_cell/split_1/split_dim
gru/cell/gru_cell/split_1SplitV$gru/cell/gru_cell/BiasAdd_1:output:0"gru/cell/gru_cell/Const_1:output:0,gru/cell/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2
gru/cell/gru_cell/split_1§
gru/cell/gru_cell/addAddV2 gru/cell/gru_cell/split:output:0"gru/cell/gru_cell/split_1:output:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/add
gru/cell/gru_cell/SigmoidSigmoidgru/cell/gru_cell/add:z:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/Sigmoid«
gru/cell/gru_cell/add_1AddV2 gru/cell/gru_cell/split:output:1"gru/cell/gru_cell/split_1:output:1*
T0*
_output_shapes
:	2
gru/cell/gru_cell/add_1
gru/cell/gru_cell/Sigmoid_1Sigmoidgru/cell/gru_cell/add_1:z:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/Sigmoid_1€
gru/cell/gru_cell/mulMulgru/cell/gru_cell/Sigmoid_1:y:0"gru/cell/gru_cell/split_1:output:2*
T0*
_output_shapes
:	2
gru/cell/gru_cell/mul’
gru/cell/gru_cell/add_2AddV2 gru/cell/gru_cell/split:output:2gru/cell/gru_cell/mul:z:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/add_2
gru/cell/gru_cell/TanhTanhgru/cell/gru_cell/add_2:z:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/TanhΝ
&gru/cell/gru_cell/mul_1/ReadVariableOpReadVariableOp;gru_cell_gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype02(
&gru/cell/gru_cell/mul_1/ReadVariableOp²
gru/cell/gru_cell/mul_1Mulgru/cell/gru_cell/Sigmoid:y:0.gru/cell/gru_cell/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/mul_1w
gru/cell/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru/cell/gru_cell/sub/x 
gru/cell/gru_cell/subSub gru/cell/gru_cell/sub/x:output:0gru/cell/gru_cell/Sigmoid:y:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/sub
gru/cell/gru_cell/mul_2Mulgru/cell/gru_cell/sub:z:0gru/cell/gru_cell/Tanh:y:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/mul_2
gru/cell/gru_cell/add_3AddV2gru/cell/gru_cell/mul_1:z:0gru/cell/gru_cell/mul_2:z:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/add_3‘
&gru/cell/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2(
&gru/cell/TensorArrayV2_1/element_shapeΪ
gru/cell/TensorArrayV2_1TensorListReserve/gru/cell/TensorArrayV2_1/element_shape:output:0gru/cell/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru/cell/TensorArrayV2_1`
gru/cell/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
gru/cell/time―
gru/cell/ReadVariableOpReadVariableOp;gru_cell_gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype02
gru/cell/ReadVariableOp
!gru/cell/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!gru/cell/while/maximum_iterations|
gru/cell/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru/cell/while/loop_counterΛ
gru/cell/whileWhile$gru/cell/while/loop_counter:output:0*gru/cell/while/maximum_iterations:output:0gru/cell/time:output:0!gru/cell/TensorArrayV2_1:handle:0gru/cell/ReadVariableOp:value:0gru/cell/strided_slice:output:0@gru/cell/TensorArrayUnstack/TensorListFromTensor:output_handle:07gru_cell_gru_cell_readvariableop_gru_cell_gru_cell_bias@gru_cell_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernelNgru_cell_gru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*1
_output_shapes
: : : : :	: : : : : *%
_read_only_resource_inputs
	*$
bodyR
gru_cell_while_body_2429*$
condR
gru_cell_while_cond_2428*0
output_shapes
: : : : :	: : : : : *
parallel_iterations 2
gru/cell/whileΗ
9gru/cell/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2;
9gru/cell/TensorArrayV2Stack/TensorListStack/element_shape
+gru/cell/TensorArrayV2Stack/TensorListStackTensorListStackgru/cell/while:output:3Bgru/cell/TensorArrayV2Stack/TensorListStack/element_shape:output:0*#
_output_shapes
:+*
element_dtype02-
+gru/cell/TensorArrayV2Stack/TensorListStack
gru/cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2 
gru/cell/strided_slice_2/stack
 gru/cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru/cell/strided_slice_2/stack_1
 gru/cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru/cell/strided_slice_2/stack_2Θ
gru/cell/strided_slice_2StridedSlice4gru/cell/TensorArrayV2Stack/TensorListStack:tensor:0'gru/cell/strided_slice_2/stack:output:0)gru/cell/strided_slice_2/stack_1:output:0)gru/cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	*
shrink_axis_mask2
gru/cell/strided_slice_2
gru/cell/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/cell/transpose_1/permΑ
gru/cell/transpose_1	Transpose4gru/cell/TensorArrayV2Stack/TensorListStack:tensor:0"gru/cell/transpose_1/perm:output:0*
T0*#
_output_shapes
:+2
gru/cell/transpose_1x
gru/cell/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru/cell/runtime²
gru/cell/AssignVariableOpAssignVariableOp;gru_cell_gru_cell_matmul_1_readvariableop_gru_cell_variablegru/cell/while:output:4^gru/cell/ReadVariableOp*^gru/cell/gru_cell/MatMul_1/ReadVariableOp'^gru/cell/gru_cell/mul_1/ReadVariableOp*
_output_shapes
 *
dtype02
gru/cell/AssignVariableOpj
gru/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
gru/ExpandDims/dim
gru/ExpandDims
ExpandDims!gru/cell/strided_slice_2:output:0gru/ExpandDims/dim:output:0*
T0*#
_output_shapes
:2
gru/ExpandDims
stream_2/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
stream_2/flatten/Const£
stream_2/flatten/ReshapeReshapegru/ExpandDims:output:0stream_2/flatten/Const:output:0*
T0*
_output_shapes
:	2
stream_2/flatten/Reshape}
dropout/IdentityIdentity!stream_2/flatten/Reshape:output:0*
T0*
_output_shapes
:	2
dropout/Identity₯
dense/MatMul/ReadVariableOpReadVariableOp(dense_matmul_readvariableop_dense_kernel* 
_output_shapes
:
*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMuldropout/Identity:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense/MatMul‘
dense/BiasAdd/ReadVariableOpReadVariableOp'dense_biasadd_readvariableop_dense_bias*
_output_shapes	
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense/BiasAdd­
dense_1/MatMul/ReadVariableOpReadVariableOp,dense_1_matmul_readvariableop_dense_1_kernel* 
_output_shapes
:
*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldense/BiasAdd:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense_1/MatMul©
dense_1/BiasAdd/ReadVariableOpReadVariableOp+dense_1_biasadd_readvariableop_dense_1_bias*
_output_shapes	
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense_1/BiasAddh
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*
_output_shapes
:	2
dense_1/Relu¬
dense_2/MatMul/ReadVariableOpReadVariableOp,dense_2_matmul_readvariableop_dense_2_kernel*
_output_shapes
:	*
dtype02
dense_2/MatMul/ReadVariableOp
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_2/MatMul¨
dense_2/BiasAdd/ReadVariableOpReadVariableOp+dense_2_biasadd_readvariableop_dense_2_bias*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_2/BiasAdd
IdentityIdentitydense_2/BiasAdd:output:0^gru/cell/AssignVariableOp^gru/cell/while*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:1(::::::::::::::26
gru/cell/AssignVariableOpgru/cell/AssignVariableOp2 
gru/cell/whilegru/cell/while:S O
+
_output_shapes
:?????????1(
 
_user_specified_nameinputs
Ψ

`
A__inference_dropout_layer_call_and_return_conditional_losses_3434

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *δ8?2
dropout/Constk
dropout/MulMulinputsdropout/Const:output:0*
T0*
_output_shapes
:	2
dropout/Mulo
dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2
dropout/Shape¬
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
_output_shapes
:	*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜΜ=2
dropout/GreaterEqual/yΆ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	2
dropout/GreaterEqualw
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	2
dropout/Castr
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*
_output_shapes
:	2
dropout/Mul_1]
IdentityIdentitydropout/Mul_1:z:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs
§
?
F__inference_functional_1_layer_call_and_return_conditional_losses_2986
input_1<
8stream_conv2d_conv2d_readvariableop_stream_conv2d_kernel;
7stream_conv2d_biasadd_readvariableop_stream_conv2d_biasD
@stream_1_conv2d_1_conv2d_readvariableop_stream_1_conv2d_1_kernelC
?stream_1_conv2d_1_biasadd_readvariableop_stream_1_conv2d_1_bias;
7gru_cell_gru_cell_readvariableop_gru_cell_gru_cell_biasD
@gru_cell_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel?
;gru_cell_gru_cell_matmul_1_readvariableop_gru_cell_variableR
Ngru_cell_gru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel,
(dense_matmul_readvariableop_dense_kernel+
'dense_biasadd_readvariableop_dense_bias0
,dense_1_matmul_readvariableop_dense_1_kernel/
+dense_1_biasadd_readvariableop_dense_1_bias0
,dense_2_matmul_readvariableop_dense_2_kernel/
+dense_2_biasadd_readvariableop_dense_2_bias
identity’gru/cell/AssignVariableOp’gru/cell/while
%tf_op_layer_ExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%tf_op_layer_ExpandDims/ExpandDims/dimΝ
!tf_op_layer_ExpandDims/ExpandDims
ExpandDimsinput_1.tf_op_layer_ExpandDims/ExpandDims/dim:output:0*
T0*
_cloned(*&
_output_shapes
:1(2#
!tf_op_layer_ExpandDims/ExpandDimsΛ
#stream/conv2d/Conv2D/ReadVariableOpReadVariableOp8stream_conv2d_conv2d_readvariableop_stream_conv2d_kernel*&
_output_shapes
:*
dtype02%
#stream/conv2d/Conv2D/ReadVariableOpι
stream/conv2d/Conv2DConv2D*tf_op_layer_ExpandDims/ExpandDims:output:0+stream/conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:/&*
paddingVALID*
strides
2
stream/conv2d/Conv2Dΐ
$stream/conv2d/BiasAdd/ReadVariableOpReadVariableOp7stream_conv2d_biasadd_readvariableop_stream_conv2d_bias*
_output_shapes
:*
dtype02&
$stream/conv2d/BiasAdd/ReadVariableOp·
stream/conv2d/BiasAddBiasAddstream/conv2d/Conv2D:output:0,stream/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:/&2
stream/conv2d/BiasAdd
stream/conv2d/ReluRelustream/conv2d/BiasAdd:output:0*
T0*&
_output_shapes
:/&2
stream/conv2d/ReluΫ
'stream_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp@stream_1_conv2d_1_conv2d_readvariableop_stream_1_conv2d_1_kernel*&
_output_shapes
:*
dtype02)
'stream_1/conv2d_1/Conv2D/ReadVariableOpλ
stream_1/conv2d_1/Conv2DConv2D stream/conv2d/Relu:activations:0/stream_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:+$*
paddingVALID*
strides
2
stream_1/conv2d_1/Conv2DΠ
(stream_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp?stream_1_conv2d_1_biasadd_readvariableop_stream_1_conv2d_1_bias*
_output_shapes
:*
dtype02*
(stream_1/conv2d_1/BiasAdd/ReadVariableOpΗ
stream_1/conv2d_1/BiasAddBiasAdd!stream_1/conv2d_1/Conv2D:output:00stream_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:+$2
stream_1/conv2d_1/BiasAdd
stream_1/conv2d_1/ReluRelu"stream_1/conv2d_1/BiasAdd:output:0*
T0*&
_output_shapes
:+$2
stream_1/conv2d_1/Reluw
reshape/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"   +   $      2
reshape/Shape
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slice}
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
reshape/Reshape/shape/1u
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :ΐ2
reshape/Reshape/shape/2Θ
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape‘
reshape/ReshapeReshape$stream_1/conv2d_1/Relu:activations:0reshape/Reshape/shape:output:0*
T0*#
_output_shapes
:+ΐ2
reshape/Reshape
gru/cell/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/cell/transpose/perm
gru/cell/transpose	Transposereshape/Reshape:output:0 gru/cell/transpose/perm:output:0*
T0*#
_output_shapes
:+ΐ2
gru/cell/transposeu
gru/cell/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"+      @  2
gru/cell/Shape
gru/cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/cell/strided_slice/stack
gru/cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru/cell/strided_slice/stack_1
gru/cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru/cell/strided_slice/stack_2
gru/cell/strided_sliceStridedSlicegru/cell/Shape:output:0%gru/cell/strided_slice/stack:output:0'gru/cell/strided_slice/stack_1:output:0'gru/cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru/cell/strided_slice
$gru/cell/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$gru/cell/TensorArrayV2/element_shapeΤ
gru/cell/TensorArrayV2TensorListReserve-gru/cell/TensorArrayV2/element_shape:output:0gru/cell/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru/cell/TensorArrayV2Ρ
>gru/cell/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   @  2@
>gru/cell/TensorArrayUnstack/TensorListFromTensor/element_shape
0gru/cell/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru/cell/transpose:y:0Ggru/cell/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type022
0gru/cell/TensorArrayUnstack/TensorListFromTensor
gru/cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru/cell/strided_slice_1/stack
 gru/cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru/cell/strided_slice_1/stack_1
 gru/cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru/cell/strided_slice_1/stack_2ͺ
gru/cell/strided_slice_1StridedSlicegru/cell/transpose:y:0'gru/cell/strided_slice_1/stack:output:0)gru/cell/strided_slice_1/stack_1:output:0)gru/cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ΐ*
shrink_axis_mask2
gru/cell/strided_slice_1½
 gru/cell/gru_cell/ReadVariableOpReadVariableOp7gru_cell_gru_cell_readvariableop_gru_cell_gru_cell_bias*
_output_shapes
:	*
dtype02"
 gru/cell/gru_cell/ReadVariableOp’
gru/cell/gru_cell/unstackUnpack(gru/cell/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
gru/cell/gru_cell/unstackΥ
'gru/cell/gru_cell/MatMul/ReadVariableOpReadVariableOp@gru_cell_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel* 
_output_shapes
:
ΐ*
dtype02)
'gru/cell/gru_cell/MatMul/ReadVariableOpΌ
gru/cell/gru_cell/MatMulMatMul!gru/cell/strided_slice_1:output:0/gru/cell/gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/MatMul³
gru/cell/gru_cell/BiasAddBiasAdd"gru/cell/gru_cell/MatMul:product:0"gru/cell/gru_cell/unstack:output:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/BiasAddt
gru/cell/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru/cell/gru_cell/Const
!gru/cell/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!gru/cell/gru_cell/split/split_dimά
gru/cell/gru_cell/splitSplit*gru/cell/gru_cell/split/split_dim:output:0"gru/cell/gru_cell/BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
gru/cell/gru_cell/splitΣ
)gru/cell/gru_cell/MatMul_1/ReadVariableOpReadVariableOp;gru_cell_gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype02+
)gru/cell/gru_cell/MatMul_1/ReadVariableOpλ
+gru/cell/gru_cell/MatMul_1/ReadVariableOp_1ReadVariableOpNgru_cell_gru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel* 
_output_shapes
:
*
dtype02-
+gru/cell/gru_cell/MatMul_1/ReadVariableOp_1Τ
gru/cell/gru_cell/MatMul_1MatMul1gru/cell/gru_cell/MatMul_1/ReadVariableOp:value:03gru/cell/gru_cell/MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/MatMul_1Ή
gru/cell/gru_cell/BiasAdd_1BiasAdd$gru/cell/gru_cell/MatMul_1:product:0"gru/cell/gru_cell/unstack:output:1*
T0*
_output_shapes
:	2
gru/cell/gru_cell/BiasAdd_1
gru/cell/gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ????2
gru/cell/gru_cell/Const_1
#gru/cell/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#gru/cell/gru_cell/split_1/split_dim
gru/cell/gru_cell/split_1SplitV$gru/cell/gru_cell/BiasAdd_1:output:0"gru/cell/gru_cell/Const_1:output:0,gru/cell/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2
gru/cell/gru_cell/split_1§
gru/cell/gru_cell/addAddV2 gru/cell/gru_cell/split:output:0"gru/cell/gru_cell/split_1:output:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/add
gru/cell/gru_cell/SigmoidSigmoidgru/cell/gru_cell/add:z:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/Sigmoid«
gru/cell/gru_cell/add_1AddV2 gru/cell/gru_cell/split:output:1"gru/cell/gru_cell/split_1:output:1*
T0*
_output_shapes
:	2
gru/cell/gru_cell/add_1
gru/cell/gru_cell/Sigmoid_1Sigmoidgru/cell/gru_cell/add_1:z:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/Sigmoid_1€
gru/cell/gru_cell/mulMulgru/cell/gru_cell/Sigmoid_1:y:0"gru/cell/gru_cell/split_1:output:2*
T0*
_output_shapes
:	2
gru/cell/gru_cell/mul’
gru/cell/gru_cell/add_2AddV2 gru/cell/gru_cell/split:output:2gru/cell/gru_cell/mul:z:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/add_2
gru/cell/gru_cell/TanhTanhgru/cell/gru_cell/add_2:z:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/TanhΝ
&gru/cell/gru_cell/mul_1/ReadVariableOpReadVariableOp;gru_cell_gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype02(
&gru/cell/gru_cell/mul_1/ReadVariableOp²
gru/cell/gru_cell/mul_1Mulgru/cell/gru_cell/Sigmoid:y:0.gru/cell/gru_cell/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/mul_1w
gru/cell/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru/cell/gru_cell/sub/x 
gru/cell/gru_cell/subSub gru/cell/gru_cell/sub/x:output:0gru/cell/gru_cell/Sigmoid:y:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/sub
gru/cell/gru_cell/mul_2Mulgru/cell/gru_cell/sub:z:0gru/cell/gru_cell/Tanh:y:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/mul_2
gru/cell/gru_cell/add_3AddV2gru/cell/gru_cell/mul_1:z:0gru/cell/gru_cell/mul_2:z:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/add_3‘
&gru/cell/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2(
&gru/cell/TensorArrayV2_1/element_shapeΪ
gru/cell/TensorArrayV2_1TensorListReserve/gru/cell/TensorArrayV2_1/element_shape:output:0gru/cell/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru/cell/TensorArrayV2_1`
gru/cell/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
gru/cell/time―
gru/cell/ReadVariableOpReadVariableOp;gru_cell_gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype02
gru/cell/ReadVariableOp
!gru/cell/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!gru/cell/while/maximum_iterations|
gru/cell/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru/cell/while/loop_counterΛ
gru/cell/whileWhile$gru/cell/while/loop_counter:output:0*gru/cell/while/maximum_iterations:output:0gru/cell/time:output:0!gru/cell/TensorArrayV2_1:handle:0gru/cell/ReadVariableOp:value:0gru/cell/strided_slice:output:0@gru/cell/TensorArrayUnstack/TensorListFromTensor:output_handle:07gru_cell_gru_cell_readvariableop_gru_cell_gru_cell_bias@gru_cell_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernelNgru_cell_gru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*1
_output_shapes
: : : : :	: : : : : *%
_read_only_resource_inputs
	*$
bodyR
gru_cell_while_body_2872*$
condR
gru_cell_while_cond_2871*0
output_shapes
: : : : :	: : : : : *
parallel_iterations 2
gru/cell/whileΗ
9gru/cell/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2;
9gru/cell/TensorArrayV2Stack/TensorListStack/element_shape
+gru/cell/TensorArrayV2Stack/TensorListStackTensorListStackgru/cell/while:output:3Bgru/cell/TensorArrayV2Stack/TensorListStack/element_shape:output:0*#
_output_shapes
:+*
element_dtype02-
+gru/cell/TensorArrayV2Stack/TensorListStack
gru/cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2 
gru/cell/strided_slice_2/stack
 gru/cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru/cell/strided_slice_2/stack_1
 gru/cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru/cell/strided_slice_2/stack_2Θ
gru/cell/strided_slice_2StridedSlice4gru/cell/TensorArrayV2Stack/TensorListStack:tensor:0'gru/cell/strided_slice_2/stack:output:0)gru/cell/strided_slice_2/stack_1:output:0)gru/cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	*
shrink_axis_mask2
gru/cell/strided_slice_2
gru/cell/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/cell/transpose_1/permΑ
gru/cell/transpose_1	Transpose4gru/cell/TensorArrayV2Stack/TensorListStack:tensor:0"gru/cell/transpose_1/perm:output:0*
T0*#
_output_shapes
:+2
gru/cell/transpose_1x
gru/cell/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru/cell/runtime²
gru/cell/AssignVariableOpAssignVariableOp;gru_cell_gru_cell_matmul_1_readvariableop_gru_cell_variablegru/cell/while:output:4^gru/cell/ReadVariableOp*^gru/cell/gru_cell/MatMul_1/ReadVariableOp'^gru/cell/gru_cell/mul_1/ReadVariableOp*
_output_shapes
 *
dtype02
gru/cell/AssignVariableOpj
gru/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
gru/ExpandDims/dim
gru/ExpandDims
ExpandDims!gru/cell/strided_slice_2:output:0gru/ExpandDims/dim:output:0*
T0*#
_output_shapes
:2
gru/ExpandDims
stream_2/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
stream_2/flatten/Const£
stream_2/flatten/ReshapeReshapegru/ExpandDims:output:0stream_2/flatten/Const:output:0*
T0*
_output_shapes
:	2
stream_2/flatten/Reshape}
dropout/IdentityIdentity!stream_2/flatten/Reshape:output:0*
T0*
_output_shapes
:	2
dropout/Identity₯
dense/MatMul/ReadVariableOpReadVariableOp(dense_matmul_readvariableop_dense_kernel* 
_output_shapes
:
*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMuldropout/Identity:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense/MatMul‘
dense/BiasAdd/ReadVariableOpReadVariableOp'dense_biasadd_readvariableop_dense_bias*
_output_shapes	
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense/BiasAdd­
dense_1/MatMul/ReadVariableOpReadVariableOp,dense_1_matmul_readvariableop_dense_1_kernel* 
_output_shapes
:
*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldense/BiasAdd:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense_1/MatMul©
dense_1/BiasAdd/ReadVariableOpReadVariableOp+dense_1_biasadd_readvariableop_dense_1_bias*
_output_shapes	
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense_1/BiasAddh
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*
_output_shapes
:	2
dense_1/Relu¬
dense_2/MatMul/ReadVariableOpReadVariableOp,dense_2_matmul_readvariableop_dense_2_kernel*
_output_shapes
:	*
dtype02
dense_2/MatMul/ReadVariableOp
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_2/MatMul¨
dense_2/BiasAdd/ReadVariableOpReadVariableOp+dense_2_biasadd_readvariableop_dense_2_bias*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_2/BiasAdd
IdentityIdentitydense_2/BiasAdd:output:0^gru/cell/AssignVariableOp^gru/cell/while*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:1(::::::::::::::26
gru/cell/AssignVariableOpgru/cell/AssignVariableOp2 
gru/cell/whilegru/cell/while:T P
+
_output_shapes
:?????????1(
!
_user_specified_name	input_1
²

while_cond_1399
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice2
.while_while_cond_1399___redundant_placeholder02
.while_while_cond_1399___redundant_placeholder12
.while_while_cond_1399___redundant_placeholder22
.while_while_cond_1399___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*8
_input_shapes'
%: : : : :	: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
:
 
Ρ
F__inference_functional_1_layer_call_and_return_conditional_losses_2344

inputs<
8stream_conv2d_conv2d_readvariableop_stream_conv2d_kernel;
7stream_conv2d_biasadd_readvariableop_stream_conv2d_biasD
@stream_1_conv2d_1_conv2d_readvariableop_stream_1_conv2d_1_kernelC
?stream_1_conv2d_1_biasadd_readvariableop_stream_1_conv2d_1_bias;
7gru_cell_gru_cell_readvariableop_gru_cell_gru_cell_biasD
@gru_cell_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel?
;gru_cell_gru_cell_matmul_1_readvariableop_gru_cell_variableR
Ngru_cell_gru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel,
(dense_matmul_readvariableop_dense_kernel+
'dense_biasadd_readvariableop_dense_bias0
,dense_1_matmul_readvariableop_dense_1_kernel/
+dense_1_biasadd_readvariableop_dense_1_bias0
,dense_2_matmul_readvariableop_dense_2_kernel/
+dense_2_biasadd_readvariableop_dense_2_bias
identity’gru/cell/AssignVariableOp’gru/cell/while
%tf_op_layer_ExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%tf_op_layer_ExpandDims/ExpandDims/dimΜ
!tf_op_layer_ExpandDims/ExpandDims
ExpandDimsinputs.tf_op_layer_ExpandDims/ExpandDims/dim:output:0*
T0*
_cloned(*&
_output_shapes
:1(2#
!tf_op_layer_ExpandDims/ExpandDimsΛ
#stream/conv2d/Conv2D/ReadVariableOpReadVariableOp8stream_conv2d_conv2d_readvariableop_stream_conv2d_kernel*&
_output_shapes
:*
dtype02%
#stream/conv2d/Conv2D/ReadVariableOpι
stream/conv2d/Conv2DConv2D*tf_op_layer_ExpandDims/ExpandDims:output:0+stream/conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:/&*
paddingVALID*
strides
2
stream/conv2d/Conv2Dΐ
$stream/conv2d/BiasAdd/ReadVariableOpReadVariableOp7stream_conv2d_biasadd_readvariableop_stream_conv2d_bias*
_output_shapes
:*
dtype02&
$stream/conv2d/BiasAdd/ReadVariableOp·
stream/conv2d/BiasAddBiasAddstream/conv2d/Conv2D:output:0,stream/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:/&2
stream/conv2d/BiasAdd
stream/conv2d/ReluRelustream/conv2d/BiasAdd:output:0*
T0*&
_output_shapes
:/&2
stream/conv2d/ReluΫ
'stream_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp@stream_1_conv2d_1_conv2d_readvariableop_stream_1_conv2d_1_kernel*&
_output_shapes
:*
dtype02)
'stream_1/conv2d_1/Conv2D/ReadVariableOpλ
stream_1/conv2d_1/Conv2DConv2D stream/conv2d/Relu:activations:0/stream_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:+$*
paddingVALID*
strides
2
stream_1/conv2d_1/Conv2DΠ
(stream_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp?stream_1_conv2d_1_biasadd_readvariableop_stream_1_conv2d_1_bias*
_output_shapes
:*
dtype02*
(stream_1/conv2d_1/BiasAdd/ReadVariableOpΗ
stream_1/conv2d_1/BiasAddBiasAdd!stream_1/conv2d_1/Conv2D:output:00stream_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:+$2
stream_1/conv2d_1/BiasAdd
stream_1/conv2d_1/ReluRelu"stream_1/conv2d_1/BiasAdd:output:0*
T0*&
_output_shapes
:+$2
stream_1/conv2d_1/Reluw
reshape/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"   +   $      2
reshape/Shape
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slice}
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
reshape/Reshape/shape/1u
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :ΐ2
reshape/Reshape/shape/2Θ
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape‘
reshape/ReshapeReshape$stream_1/conv2d_1/Relu:activations:0reshape/Reshape/shape:output:0*
T0*#
_output_shapes
:+ΐ2
reshape/Reshape
gru/cell/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/cell/transpose/perm
gru/cell/transpose	Transposereshape/Reshape:output:0 gru/cell/transpose/perm:output:0*
T0*#
_output_shapes
:+ΐ2
gru/cell/transposeu
gru/cell/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"+      @  2
gru/cell/Shape
gru/cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/cell/strided_slice/stack
gru/cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru/cell/strided_slice/stack_1
gru/cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru/cell/strided_slice/stack_2
gru/cell/strided_sliceStridedSlicegru/cell/Shape:output:0%gru/cell/strided_slice/stack:output:0'gru/cell/strided_slice/stack_1:output:0'gru/cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru/cell/strided_slice
$gru/cell/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$gru/cell/TensorArrayV2/element_shapeΤ
gru/cell/TensorArrayV2TensorListReserve-gru/cell/TensorArrayV2/element_shape:output:0gru/cell/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru/cell/TensorArrayV2Ρ
>gru/cell/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   @  2@
>gru/cell/TensorArrayUnstack/TensorListFromTensor/element_shape
0gru/cell/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru/cell/transpose:y:0Ggru/cell/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type022
0gru/cell/TensorArrayUnstack/TensorListFromTensor
gru/cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru/cell/strided_slice_1/stack
 gru/cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru/cell/strided_slice_1/stack_1
 gru/cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru/cell/strided_slice_1/stack_2ͺ
gru/cell/strided_slice_1StridedSlicegru/cell/transpose:y:0'gru/cell/strided_slice_1/stack:output:0)gru/cell/strided_slice_1/stack_1:output:0)gru/cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ΐ*
shrink_axis_mask2
gru/cell/strided_slice_1½
 gru/cell/gru_cell/ReadVariableOpReadVariableOp7gru_cell_gru_cell_readvariableop_gru_cell_gru_cell_bias*
_output_shapes
:	*
dtype02"
 gru/cell/gru_cell/ReadVariableOp’
gru/cell/gru_cell/unstackUnpack(gru/cell/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
gru/cell/gru_cell/unstackΥ
'gru/cell/gru_cell/MatMul/ReadVariableOpReadVariableOp@gru_cell_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel* 
_output_shapes
:
ΐ*
dtype02)
'gru/cell/gru_cell/MatMul/ReadVariableOpΌ
gru/cell/gru_cell/MatMulMatMul!gru/cell/strided_slice_1:output:0/gru/cell/gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/MatMul³
gru/cell/gru_cell/BiasAddBiasAdd"gru/cell/gru_cell/MatMul:product:0"gru/cell/gru_cell/unstack:output:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/BiasAddt
gru/cell/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru/cell/gru_cell/Const
!gru/cell/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!gru/cell/gru_cell/split/split_dimά
gru/cell/gru_cell/splitSplit*gru/cell/gru_cell/split/split_dim:output:0"gru/cell/gru_cell/BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
gru/cell/gru_cell/splitΣ
)gru/cell/gru_cell/MatMul_1/ReadVariableOpReadVariableOp;gru_cell_gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype02+
)gru/cell/gru_cell/MatMul_1/ReadVariableOpλ
+gru/cell/gru_cell/MatMul_1/ReadVariableOp_1ReadVariableOpNgru_cell_gru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel* 
_output_shapes
:
*
dtype02-
+gru/cell/gru_cell/MatMul_1/ReadVariableOp_1Τ
gru/cell/gru_cell/MatMul_1MatMul1gru/cell/gru_cell/MatMul_1/ReadVariableOp:value:03gru/cell/gru_cell/MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/MatMul_1Ή
gru/cell/gru_cell/BiasAdd_1BiasAdd$gru/cell/gru_cell/MatMul_1:product:0"gru/cell/gru_cell/unstack:output:1*
T0*
_output_shapes
:	2
gru/cell/gru_cell/BiasAdd_1
gru/cell/gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ????2
gru/cell/gru_cell/Const_1
#gru/cell/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#gru/cell/gru_cell/split_1/split_dim
gru/cell/gru_cell/split_1SplitV$gru/cell/gru_cell/BiasAdd_1:output:0"gru/cell/gru_cell/Const_1:output:0,gru/cell/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2
gru/cell/gru_cell/split_1§
gru/cell/gru_cell/addAddV2 gru/cell/gru_cell/split:output:0"gru/cell/gru_cell/split_1:output:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/add
gru/cell/gru_cell/SigmoidSigmoidgru/cell/gru_cell/add:z:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/Sigmoid«
gru/cell/gru_cell/add_1AddV2 gru/cell/gru_cell/split:output:1"gru/cell/gru_cell/split_1:output:1*
T0*
_output_shapes
:	2
gru/cell/gru_cell/add_1
gru/cell/gru_cell/Sigmoid_1Sigmoidgru/cell/gru_cell/add_1:z:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/Sigmoid_1€
gru/cell/gru_cell/mulMulgru/cell/gru_cell/Sigmoid_1:y:0"gru/cell/gru_cell/split_1:output:2*
T0*
_output_shapes
:	2
gru/cell/gru_cell/mul’
gru/cell/gru_cell/add_2AddV2 gru/cell/gru_cell/split:output:2gru/cell/gru_cell/mul:z:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/add_2
gru/cell/gru_cell/TanhTanhgru/cell/gru_cell/add_2:z:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/TanhΝ
&gru/cell/gru_cell/mul_1/ReadVariableOpReadVariableOp;gru_cell_gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype02(
&gru/cell/gru_cell/mul_1/ReadVariableOp²
gru/cell/gru_cell/mul_1Mulgru/cell/gru_cell/Sigmoid:y:0.gru/cell/gru_cell/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/mul_1w
gru/cell/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru/cell/gru_cell/sub/x 
gru/cell/gru_cell/subSub gru/cell/gru_cell/sub/x:output:0gru/cell/gru_cell/Sigmoid:y:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/sub
gru/cell/gru_cell/mul_2Mulgru/cell/gru_cell/sub:z:0gru/cell/gru_cell/Tanh:y:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/mul_2
gru/cell/gru_cell/add_3AddV2gru/cell/gru_cell/mul_1:z:0gru/cell/gru_cell/mul_2:z:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/add_3‘
&gru/cell/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2(
&gru/cell/TensorArrayV2_1/element_shapeΪ
gru/cell/TensorArrayV2_1TensorListReserve/gru/cell/TensorArrayV2_1/element_shape:output:0gru/cell/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru/cell/TensorArrayV2_1`
gru/cell/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
gru/cell/time―
gru/cell/ReadVariableOpReadVariableOp;gru_cell_gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype02
gru/cell/ReadVariableOp
!gru/cell/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!gru/cell/while/maximum_iterations|
gru/cell/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru/cell/while/loop_counterΛ
gru/cell/whileWhile$gru/cell/while/loop_counter:output:0*gru/cell/while/maximum_iterations:output:0gru/cell/time:output:0!gru/cell/TensorArrayV2_1:handle:0gru/cell/ReadVariableOp:value:0gru/cell/strided_slice:output:0@gru/cell/TensorArrayUnstack/TensorListFromTensor:output_handle:07gru_cell_gru_cell_readvariableop_gru_cell_gru_cell_bias@gru_cell_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernelNgru_cell_gru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*1
_output_shapes
: : : : :	: : : : : *%
_read_only_resource_inputs
	*$
bodyR
gru_cell_while_body_2223*$
condR
gru_cell_while_cond_2222*0
output_shapes
: : : : :	: : : : : *
parallel_iterations 2
gru/cell/whileΗ
9gru/cell/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2;
9gru/cell/TensorArrayV2Stack/TensorListStack/element_shape
+gru/cell/TensorArrayV2Stack/TensorListStackTensorListStackgru/cell/while:output:3Bgru/cell/TensorArrayV2Stack/TensorListStack/element_shape:output:0*#
_output_shapes
:+*
element_dtype02-
+gru/cell/TensorArrayV2Stack/TensorListStack
gru/cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2 
gru/cell/strided_slice_2/stack
 gru/cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru/cell/strided_slice_2/stack_1
 gru/cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru/cell/strided_slice_2/stack_2Θ
gru/cell/strided_slice_2StridedSlice4gru/cell/TensorArrayV2Stack/TensorListStack:tensor:0'gru/cell/strided_slice_2/stack:output:0)gru/cell/strided_slice_2/stack_1:output:0)gru/cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	*
shrink_axis_mask2
gru/cell/strided_slice_2
gru/cell/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/cell/transpose_1/permΑ
gru/cell/transpose_1	Transpose4gru/cell/TensorArrayV2Stack/TensorListStack:tensor:0"gru/cell/transpose_1/perm:output:0*
T0*#
_output_shapes
:+2
gru/cell/transpose_1x
gru/cell/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru/cell/runtime²
gru/cell/AssignVariableOpAssignVariableOp;gru_cell_gru_cell_matmul_1_readvariableop_gru_cell_variablegru/cell/while:output:4^gru/cell/ReadVariableOp*^gru/cell/gru_cell/MatMul_1/ReadVariableOp'^gru/cell/gru_cell/mul_1/ReadVariableOp*
_output_shapes
 *
dtype02
gru/cell/AssignVariableOpj
gru/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
gru/ExpandDims/dim
gru/ExpandDims
ExpandDims!gru/cell/strided_slice_2:output:0gru/ExpandDims/dim:output:0*
T0*#
_output_shapes
:2
gru/ExpandDims
stream_2/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
stream_2/flatten/Const£
stream_2/flatten/ReshapeReshapegru/ExpandDims:output:0stream_2/flatten/Const:output:0*
T0*
_output_shapes
:	2
stream_2/flatten/Reshapes
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *δ8?2
dropout/dropout/Const
dropout/dropout/MulMul!stream_2/flatten/Reshape:output:0dropout/dropout/Const:output:0*
T0*
_output_shapes
:	2
dropout/dropout/Mul
dropout/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2
dropout/dropout/ShapeΔ
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*
_output_shapes
:	*
dtype02.
,dropout/dropout/random_uniform/RandomUniform
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜΜ=2 
dropout/dropout/GreaterEqual/yΦ
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	2
dropout/dropout/GreaterEqual
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	2
dropout/dropout/Cast
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*
_output_shapes
:	2
dropout/dropout/Mul_1₯
dense/MatMul/ReadVariableOpReadVariableOp(dense_matmul_readvariableop_dense_kernel* 
_output_shapes
:
*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMuldropout/dropout/Mul_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense/MatMul‘
dense/BiasAdd/ReadVariableOpReadVariableOp'dense_biasadd_readvariableop_dense_bias*
_output_shapes	
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense/BiasAdd­
dense_1/MatMul/ReadVariableOpReadVariableOp,dense_1_matmul_readvariableop_dense_1_kernel* 
_output_shapes
:
*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldense/BiasAdd:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense_1/MatMul©
dense_1/BiasAdd/ReadVariableOpReadVariableOp+dense_1_biasadd_readvariableop_dense_1_bias*
_output_shapes	
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense_1/BiasAddh
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*
_output_shapes
:	2
dense_1/Relu¬
dense_2/MatMul/ReadVariableOpReadVariableOp,dense_2_matmul_readvariableop_dense_2_kernel*
_output_shapes
:	*
dtype02
dense_2/MatMul/ReadVariableOp
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_2/MatMul¨
dense_2/BiasAdd/ReadVariableOpReadVariableOp+dense_2_biasadd_readvariableop_dense_2_bias*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_2/BiasAdd
IdentityIdentitydense_2/BiasAdd:output:0^gru/cell/AssignVariableOp^gru/cell/while*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:1(::::::::::::::26
gru/cell/AssignVariableOpgru/cell/AssignVariableOp2 
gru/cell/whilegru/cell/while:S O
+
_output_shapes
:?????????1(
 
_user_specified_nameinputs
²

while_cond_3710
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice2
.while_while_cond_3710___redundant_placeholder02
.while_while_cond_3710___redundant_placeholder12
.while_while_cond_3710___redundant_placeholder22
.while_while_cond_3710___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*8
_input_shapes'
%: : : : :	: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
:
δ(
·
__inference__traced_save_4180
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop3
/savev2_stream_conv2d_kernel_read_readvariableop1
-savev2_stream_conv2d_bias_read_readvariableop7
3savev2_stream_1_conv2d_1_kernel_read_readvariableop5
1savev2_stream_1_conv2d_1_bias_read_readvariableop7
3savev2_gru_cell_gru_cell_kernel_read_readvariableopA
=savev2_gru_cell_gru_cell_recurrent_kernel_read_readvariableop5
1savev2_gru_cell_gru_cell_bias_read_readvariableop0
,savev2_gru_cell_variable_read_readvariableop
savev2_const

identity_1’MergeV2Checkpoints
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
value3B1 B+_temp_7441c2533f7141c0a20ee5d650d2954b/part2	
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
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameΛ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*έ
valueΣBΠB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEBFlayer_with_weights-2/gru/keras_api/states/0/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names¦
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesέ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop/savev2_stream_conv2d_kernel_read_readvariableop-savev2_stream_conv2d_bias_read_readvariableop3savev2_stream_1_conv2d_1_kernel_read_readvariableop1savev2_stream_1_conv2d_1_bias_read_readvariableop3savev2_gru_cell_gru_cell_kernel_read_readvariableop=savev2_gru_cell_gru_cell_recurrent_kernel_read_readvariableop1savev2_gru_cell_gru_cell_bias_read_readvariableop,savev2_gru_cell_variable_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2Ί
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes‘
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

identity_1Identity_1:output:0*?
_input_shapes
: :
::
::	::::::
ΐ:
:	:	: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,	(
&
_output_shapes
:: 


_output_shapes
::&"
 
_output_shapes
:
ΐ:&"
 
_output_shapes
:
:%!

_output_shapes
:	:%!

_output_shapes
:	:

_output_shapes
: 
Έ	
Ι
gru_cell_while_cond_2665.
*gru_cell_while_gru_cell_while_loop_counter4
0gru_cell_while_gru_cell_while_maximum_iterations
gru_cell_while_placeholder 
gru_cell_while_placeholder_1 
gru_cell_while_placeholder_2.
*gru_cell_while_less_gru_cell_strided_sliceD
@gru_cell_while_gru_cell_while_cond_2665___redundant_placeholder0D
@gru_cell_while_gru_cell_while_cond_2665___redundant_placeholder1D
@gru_cell_while_gru_cell_while_cond_2665___redundant_placeholder2D
@gru_cell_while_gru_cell_while_cond_2665___redundant_placeholder3
gru_cell_while_identity

gru/cell/while/LessLessgru_cell_while_placeholder*gru_cell_while_less_gru_cell_strided_slice*
T0*
_output_shapes
: 2
gru/cell/while/Lessx
gru/cell/while/IdentityIdentitygru/cell/while/Less:z:0*
T0
*
_output_shapes
: 2
gru/cell/while/Identity";
gru_cell_while_identity gru/cell/while/Identity:output:0*8
_input_shapes'
%: : : : :	: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
:

 
B__inference_gru_cell_layer_call_and_return_conditional_losses_3862

inputs
states_0)
%readvariableop_gru_cell_gru_cell_bias2
.matmul_readvariableop_gru_cell_gru_cell_kernel@
<matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel
identity

identity_1
ReadVariableOpReadVariableOp%readvariableop_gru_cell_gru_cell_bias*
_output_shapes
:	*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOp.matmul_readvariableop_gru_cell_gru_cell_kernel* 
_output_shapes
:
ΐ*
dtype02
MatMul/ReadVariableOpk
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
MatMulk
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*
_output_shapes
:	2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split/split_dim
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
splitu
MatMul_1/ReadVariableOpReadVariableOpstates_0*
_output_shapes
:*
dtype02
MatMul_1/ReadVariableOp΅
MatMul_1/ReadVariableOp_1ReadVariableOp<matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOp_1
MatMul_1BatchMatMulV2MatMul_1/ReadVariableOp:value:0!MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
:2

MatMul_1i
	BiasAdd_1BiasAddMatMul_1:output:0unstack:output:1*
T0*
_output_shapes
:2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ????2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split_1/split_dim¦
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0* 
_output_shapes
:::*
	num_split2	
split_1X
addAddV2split:output:0split_1:output:0*
T0*
_output_shapes
:2
addI
SigmoidSigmoidadd:z:0*
T0*
_output_shapes
:2	
Sigmoid\
add_1AddV2split:output:1split_1:output:1*
T0*
_output_shapes
:2
add_1O
	Sigmoid_1Sigmoid	add_1:z:0*
T0*
_output_shapes
:2
	Sigmoid_1U
mulMulSigmoid_1:y:0split_1:output:2*
T0*
_output_shapes
:2
mulS
add_2AddV2split:output:2mul:z:0*
T0*
_output_shapes
:2
add_2B
TanhTanh	add_2:z:0*
T0*
_output_shapes
:2
Tanho
mul_1/ReadVariableOpReadVariableOpstates_0*
_output_shapes
:*
dtype02
mul_1/ReadVariableOpc
mul_1MulSigmoid:y:0mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xQ
subSubsub/x:output:0Sigmoid:y:0*
T0*
_output_shapes
:2
subK
mul_2Mulsub:z:0Tanh:y:0*
T0*
_output_shapes
:2
mul_2P
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*
_output_shapes
:2
add_3N
IdentityIdentity	add_3:z:0*
T0*
_output_shapes
:2

IdentityR

Identity_1Identity	add_3:z:0*
T0*
_output_shapes
:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*.
_input_shapes
:	ΐ:::::G C

_output_shapes
:	ΐ
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0
Δ
ύ
cell_while_cond_3148&
"cell_while_cell_while_loop_counter,
(cell_while_cell_while_maximum_iterations
cell_while_placeholder
cell_while_placeholder_1
cell_while_placeholder_2&
"cell_while_less_cell_strided_slice<
8cell_while_cell_while_cond_3148___redundant_placeholder0<
8cell_while_cell_while_cond_3148___redundant_placeholder1<
8cell_while_cell_while_cond_3148___redundant_placeholder2<
8cell_while_cell_while_cond_3148___redundant_placeholder3
cell_while_identity

cell/while/LessLesscell_while_placeholder"cell_while_less_cell_strided_slice*
T0*
_output_shapes
: 2
cell/while/Lessl
cell/while/IdentityIdentitycell/while/Less:z:0*
T0
*
_output_shapes
: 2
cell/while/Identity"3
cell_while_identitycell/while/Identity:output:0*8
_input_shapes'
%: : : : :	: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
:


B__inference_gru_cell_layer_call_and_return_conditional_losses_4093

inputs
states_0)
%readvariableop_gru_cell_gru_cell_bias2
.matmul_readvariableop_gru_cell_gru_cell_kernel>
:matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel
identity

identity_1
ReadVariableOpReadVariableOp%readvariableop_gru_cell_gru_cell_bias*
_output_shapes
:	*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOp.matmul_readvariableop_gru_cell_gru_cell_kernel* 
_output_shapes
:
ΐ*
dtype02
MatMul/ReadVariableOpk
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
MatMulk
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*
_output_shapes
:	2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split/split_dim
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
split―
MatMul_1/ReadVariableOpReadVariableOp:matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOps
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2

MatMul_1q
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*
_output_shapes
:	2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ????2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split_1/split_dim»
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2	
split_1_
addAddV2split:output:0split_1:output:0*
T0*
_output_shapes
:	2
addP
SigmoidSigmoidadd:z:0*
T0*
_output_shapes
:	2	
Sigmoidc
add_1AddV2split:output:1split_1:output:1*
T0*
_output_shapes
:	2
add_1V
	Sigmoid_1Sigmoid	add_1:z:0*
T0*
_output_shapes
:	2
	Sigmoid_1\
mulMulSigmoid_1:y:0split_1:output:2*
T0*
_output_shapes
:	2
mulZ
add_2AddV2split:output:2mul:z:0*
T0*
_output_shapes
:	2
add_2I
TanhTanh	add_2:z:0*
T0*
_output_shapes
:	2
TanhV
mul_1MulSigmoid:y:0states_0*
T0*
_output_shapes
:	2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xX
subSubsub/x:output:0Sigmoid:y:0*
T0*
_output_shapes
:	2
subR
mul_2Mulsub:z:0Tanh:y:0*
T0*
_output_shapes
:	2
mul_2W
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*
_output_shapes
:	2
add_3U
IdentityIdentity	add_3:z:0*
T0*
_output_shapes
:	2

IdentityY

Identity_1Identity	add_3:z:0*
T0*
_output_shapes
:	2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*5
_input_shapes$
":	ΐ:	::::G C

_output_shapes
:	ΐ
 
_user_specified_nameinputs:IE

_output_shapes
:	
"
_user_specified_name
states/0
Ή

Ϊ
B__inference_stream_1_layer_call_and_return_conditional_losses_1520

inputs;
7conv2d_1_conv2d_readvariableop_stream_1_conv2d_1_kernel:
6conv2d_1_biasadd_readvariableop_stream_1_conv2d_1_bias
identityΐ
conv2d_1/Conv2D/ReadVariableOpReadVariableOp7conv2d_1_conv2d_readvariableop_stream_1_conv2d_1_kernel*&
_output_shapes
:*
dtype02 
conv2d_1/Conv2D/ReadVariableOpΆ
conv2d_1/Conv2DConv2Dinputs&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:+$*
paddingVALID*
strides
2
conv2d_1/Conv2D΅
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp6conv2d_1_biasadd_readvariableop_stream_1_conv2d_1_bias*
_output_shapes
:*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp£
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:+$2
conv2d_1/BiasAddr
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*&
_output_shapes
:+$2
conv2d_1/Relun
IdentityIdentityconv2d_1/Relu:activations:0*
T0*&
_output_shapes
:+$2

Identity"
identityIdentity:output:0*-
_input_shapes
:/&:::N J
&
_output_shapes
:/&
 
_user_specified_nameinputs
£K
	
gru_cell_while_body_2223.
*gru_cell_while_gru_cell_while_loop_counter4
0gru_cell_while_gru_cell_while_maximum_iterations
gru_cell_while_placeholder 
gru_cell_while_placeholder_1 
gru_cell_while_placeholder_2+
'gru_cell_while_gru_cell_strided_slice_0i
egru_cell_while_tensorarrayv2read_tensorlistgetitem_gru_cell_tensorarrayunstack_tensorlistfromtensor_0C
?gru_cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0L
Hgru_cell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0X
Tgru_cell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0
gru_cell_while_identity
gru_cell_while_identity_1
gru_cell_while_identity_2
gru_cell_while_identity_3
gru_cell_while_identity_4)
%gru_cell_while_gru_cell_strided_sliceg
cgru_cell_while_tensorarrayv2read_tensorlistgetitem_gru_cell_tensorarrayunstack_tensorlistfromtensorA
=gru_cell_while_gru_cell_readvariableop_gru_cell_gru_cell_biasJ
Fgru_cell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernelV
Rgru_cell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernelΥ
@gru/cell/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   @  2B
@gru/cell/while/TensorArrayV2Read/TensorListGetItem/element_shape
2gru/cell/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemegru_cell_while_tensorarrayv2read_tensorlistgetitem_gru_cell_tensorarrayunstack_tensorlistfromtensor_0gru_cell_while_placeholderIgru/cell/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes
:	ΐ*
element_dtype024
2gru/cell/while/TensorArrayV2Read/TensorListGetItemΡ
&gru/cell/while/gru_cell/ReadVariableOpReadVariableOp?gru_cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0*
_output_shapes
:	*
dtype02(
&gru/cell/while/gru_cell/ReadVariableOp΄
gru/cell/while/gru_cell/unstackUnpack.gru/cell/while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2!
gru/cell/while/gru_cell/unstackι
-gru/cell/while/gru_cell/MatMul/ReadVariableOpReadVariableOpHgru_cell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0* 
_output_shapes
:
ΐ*
dtype02/
-gru/cell/while/gru_cell/MatMul/ReadVariableOpζ
gru/cell/while/gru_cell/MatMulMatMul9gru/cell/while/TensorArrayV2Read/TensorListGetItem:item:05gru/cell/while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2 
gru/cell/while/gru_cell/MatMulΛ
gru/cell/while/gru_cell/BiasAddBiasAdd(gru/cell/while/gru_cell/MatMul:product:0(gru/cell/while/gru_cell/unstack:output:0*
T0*
_output_shapes
:	2!
gru/cell/while/gru_cell/BiasAdd
gru/cell/while/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru/cell/while/gru_cell/Const
'gru/cell/while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'gru/cell/while/gru_cell/split/split_dimτ
gru/cell/while/gru_cell/splitSplit0gru/cell/while/gru_cell/split/split_dim:output:0(gru/cell/while/gru_cell/BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
gru/cell/while/gru_cell/splitω
/gru/cell/while/gru_cell/MatMul_1/ReadVariableOpReadVariableOpTgru_cell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0* 
_output_shapes
:
*
dtype021
/gru/cell/while/gru_cell/MatMul_1/ReadVariableOpΟ
 gru/cell/while/gru_cell/MatMul_1MatMulgru_cell_while_placeholder_27gru/cell/while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2"
 gru/cell/while/gru_cell/MatMul_1Ρ
!gru/cell/while/gru_cell/BiasAdd_1BiasAdd*gru/cell/while/gru_cell/MatMul_1:product:0(gru/cell/while/gru_cell/unstack:output:1*
T0*
_output_shapes
:	2#
!gru/cell/while/gru_cell/BiasAdd_1
gru/cell/while/gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ????2!
gru/cell/while/gru_cell/Const_1‘
)gru/cell/while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2+
)gru/cell/while/gru_cell/split_1/split_dim³
gru/cell/while/gru_cell/split_1SplitV*gru/cell/while/gru_cell/BiasAdd_1:output:0(gru/cell/while/gru_cell/Const_1:output:02gru/cell/while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2!
gru/cell/while/gru_cell/split_1Ώ
gru/cell/while/gru_cell/addAddV2&gru/cell/while/gru_cell/split:output:0(gru/cell/while/gru_cell/split_1:output:0*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/add
gru/cell/while/gru_cell/SigmoidSigmoidgru/cell/while/gru_cell/add:z:0*
T0*
_output_shapes
:	2!
gru/cell/while/gru_cell/SigmoidΓ
gru/cell/while/gru_cell/add_1AddV2&gru/cell/while/gru_cell/split:output:1(gru/cell/while/gru_cell/split_1:output:1*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/add_1
!gru/cell/while/gru_cell/Sigmoid_1Sigmoid!gru/cell/while/gru_cell/add_1:z:0*
T0*
_output_shapes
:	2#
!gru/cell/while/gru_cell/Sigmoid_1Ό
gru/cell/while/gru_cell/mulMul%gru/cell/while/gru_cell/Sigmoid_1:y:0(gru/cell/while/gru_cell/split_1:output:2*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/mulΊ
gru/cell/while/gru_cell/add_2AddV2&gru/cell/while/gru_cell/split:output:2gru/cell/while/gru_cell/mul:z:0*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/add_2
gru/cell/while/gru_cell/TanhTanh!gru/cell/while/gru_cell/add_2:z:0*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/Tanh²
gru/cell/while/gru_cell/mul_1Mul#gru/cell/while/gru_cell/Sigmoid:y:0gru_cell_while_placeholder_2*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/mul_1
gru/cell/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru/cell/while/gru_cell/sub/xΈ
gru/cell/while/gru_cell/subSub&gru/cell/while/gru_cell/sub/x:output:0#gru/cell/while/gru_cell/Sigmoid:y:0*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/sub²
gru/cell/while/gru_cell/mul_2Mulgru/cell/while/gru_cell/sub:z:0 gru/cell/while/gru_cell/Tanh:y:0*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/mul_2·
gru/cell/while/gru_cell/add_3AddV2!gru/cell/while/gru_cell/mul_1:z:0!gru/cell/while/gru_cell/mul_2:z:0*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/add_3
3gru/cell/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_cell_while_placeholder_1gru_cell_while_placeholder!gru/cell/while/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype025
3gru/cell/while/TensorArrayV2Write/TensorListSetItemn
gru/cell/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru/cell/while/add/y
gru/cell/while/addAddV2gru_cell_while_placeholdergru/cell/while/add/y:output:0*
T0*
_output_shapes
: 2
gru/cell/while/addr
gru/cell/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru/cell/while/add_1/y£
gru/cell/while/add_1AddV2*gru_cell_while_gru_cell_while_loop_countergru/cell/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru/cell/while/add_1y
gru/cell/while/IdentityIdentitygru/cell/while/add_1:z:0*
T0*
_output_shapes
: 2
gru/cell/while/Identity
gru/cell/while/Identity_1Identity0gru_cell_while_gru_cell_while_maximum_iterations*
T0*
_output_shapes
: 2
gru/cell/while/Identity_1{
gru/cell/while/Identity_2Identitygru/cell/while/add:z:0*
T0*
_output_shapes
: 2
gru/cell/while/Identity_2¨
gru/cell/while/Identity_3IdentityCgru/cell/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
gru/cell/while/Identity_3
gru/cell/while/Identity_4Identity!gru/cell/while/gru_cell/add_3:z:0*
T0*
_output_shapes
:	2
gru/cell/while/Identity_4"ͺ
Rgru_cell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernelTgru_cell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0"
Fgru_cell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernelHgru_cell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0"
=gru_cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias?gru_cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0"P
%gru_cell_while_gru_cell_strided_slice'gru_cell_while_gru_cell_strided_slice_0";
gru_cell_while_identity gru/cell/while/Identity:output:0"?
gru_cell_while_identity_1"gru/cell/while/Identity_1:output:0"?
gru_cell_while_identity_2"gru/cell/while/Identity_2:output:0"?
gru_cell_while_identity_3"gru/cell/while/Identity_3:output:0"?
gru_cell_while_identity_4"gru/cell/while/Identity_4:output:0"Μ
cgru_cell_while_tensorarrayv2read_tensorlistgetitem_gru_cell_tensorarrayunstack_tensorlistfromtensoregru_cell_while_tensorarrayv2read_tensorlistgetitem_gru_cell_tensorarrayunstack_tensorlistfromtensor_0*6
_input_shapes%
#: : : : :	: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
: 
£K
	
gru_cell_while_body_2872.
*gru_cell_while_gru_cell_while_loop_counter4
0gru_cell_while_gru_cell_while_maximum_iterations
gru_cell_while_placeholder 
gru_cell_while_placeholder_1 
gru_cell_while_placeholder_2+
'gru_cell_while_gru_cell_strided_slice_0i
egru_cell_while_tensorarrayv2read_tensorlistgetitem_gru_cell_tensorarrayunstack_tensorlistfromtensor_0C
?gru_cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0L
Hgru_cell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0X
Tgru_cell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0
gru_cell_while_identity
gru_cell_while_identity_1
gru_cell_while_identity_2
gru_cell_while_identity_3
gru_cell_while_identity_4)
%gru_cell_while_gru_cell_strided_sliceg
cgru_cell_while_tensorarrayv2read_tensorlistgetitem_gru_cell_tensorarrayunstack_tensorlistfromtensorA
=gru_cell_while_gru_cell_readvariableop_gru_cell_gru_cell_biasJ
Fgru_cell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernelV
Rgru_cell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernelΥ
@gru/cell/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   @  2B
@gru/cell/while/TensorArrayV2Read/TensorListGetItem/element_shape
2gru/cell/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemegru_cell_while_tensorarrayv2read_tensorlistgetitem_gru_cell_tensorarrayunstack_tensorlistfromtensor_0gru_cell_while_placeholderIgru/cell/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes
:	ΐ*
element_dtype024
2gru/cell/while/TensorArrayV2Read/TensorListGetItemΡ
&gru/cell/while/gru_cell/ReadVariableOpReadVariableOp?gru_cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0*
_output_shapes
:	*
dtype02(
&gru/cell/while/gru_cell/ReadVariableOp΄
gru/cell/while/gru_cell/unstackUnpack.gru/cell/while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2!
gru/cell/while/gru_cell/unstackι
-gru/cell/while/gru_cell/MatMul/ReadVariableOpReadVariableOpHgru_cell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0* 
_output_shapes
:
ΐ*
dtype02/
-gru/cell/while/gru_cell/MatMul/ReadVariableOpζ
gru/cell/while/gru_cell/MatMulMatMul9gru/cell/while/TensorArrayV2Read/TensorListGetItem:item:05gru/cell/while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2 
gru/cell/while/gru_cell/MatMulΛ
gru/cell/while/gru_cell/BiasAddBiasAdd(gru/cell/while/gru_cell/MatMul:product:0(gru/cell/while/gru_cell/unstack:output:0*
T0*
_output_shapes
:	2!
gru/cell/while/gru_cell/BiasAdd
gru/cell/while/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru/cell/while/gru_cell/Const
'gru/cell/while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'gru/cell/while/gru_cell/split/split_dimτ
gru/cell/while/gru_cell/splitSplit0gru/cell/while/gru_cell/split/split_dim:output:0(gru/cell/while/gru_cell/BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
gru/cell/while/gru_cell/splitω
/gru/cell/while/gru_cell/MatMul_1/ReadVariableOpReadVariableOpTgru_cell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0* 
_output_shapes
:
*
dtype021
/gru/cell/while/gru_cell/MatMul_1/ReadVariableOpΟ
 gru/cell/while/gru_cell/MatMul_1MatMulgru_cell_while_placeholder_27gru/cell/while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2"
 gru/cell/while/gru_cell/MatMul_1Ρ
!gru/cell/while/gru_cell/BiasAdd_1BiasAdd*gru/cell/while/gru_cell/MatMul_1:product:0(gru/cell/while/gru_cell/unstack:output:1*
T0*
_output_shapes
:	2#
!gru/cell/while/gru_cell/BiasAdd_1
gru/cell/while/gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ????2!
gru/cell/while/gru_cell/Const_1‘
)gru/cell/while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2+
)gru/cell/while/gru_cell/split_1/split_dim³
gru/cell/while/gru_cell/split_1SplitV*gru/cell/while/gru_cell/BiasAdd_1:output:0(gru/cell/while/gru_cell/Const_1:output:02gru/cell/while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2!
gru/cell/while/gru_cell/split_1Ώ
gru/cell/while/gru_cell/addAddV2&gru/cell/while/gru_cell/split:output:0(gru/cell/while/gru_cell/split_1:output:0*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/add
gru/cell/while/gru_cell/SigmoidSigmoidgru/cell/while/gru_cell/add:z:0*
T0*
_output_shapes
:	2!
gru/cell/while/gru_cell/SigmoidΓ
gru/cell/while/gru_cell/add_1AddV2&gru/cell/while/gru_cell/split:output:1(gru/cell/while/gru_cell/split_1:output:1*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/add_1
!gru/cell/while/gru_cell/Sigmoid_1Sigmoid!gru/cell/while/gru_cell/add_1:z:0*
T0*
_output_shapes
:	2#
!gru/cell/while/gru_cell/Sigmoid_1Ό
gru/cell/while/gru_cell/mulMul%gru/cell/while/gru_cell/Sigmoid_1:y:0(gru/cell/while/gru_cell/split_1:output:2*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/mulΊ
gru/cell/while/gru_cell/add_2AddV2&gru/cell/while/gru_cell/split:output:2gru/cell/while/gru_cell/mul:z:0*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/add_2
gru/cell/while/gru_cell/TanhTanh!gru/cell/while/gru_cell/add_2:z:0*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/Tanh²
gru/cell/while/gru_cell/mul_1Mul#gru/cell/while/gru_cell/Sigmoid:y:0gru_cell_while_placeholder_2*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/mul_1
gru/cell/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru/cell/while/gru_cell/sub/xΈ
gru/cell/while/gru_cell/subSub&gru/cell/while/gru_cell/sub/x:output:0#gru/cell/while/gru_cell/Sigmoid:y:0*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/sub²
gru/cell/while/gru_cell/mul_2Mulgru/cell/while/gru_cell/sub:z:0 gru/cell/while/gru_cell/Tanh:y:0*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/mul_2·
gru/cell/while/gru_cell/add_3AddV2!gru/cell/while/gru_cell/mul_1:z:0!gru/cell/while/gru_cell/mul_2:z:0*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/add_3
3gru/cell/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_cell_while_placeholder_1gru_cell_while_placeholder!gru/cell/while/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype025
3gru/cell/while/TensorArrayV2Write/TensorListSetItemn
gru/cell/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru/cell/while/add/y
gru/cell/while/addAddV2gru_cell_while_placeholdergru/cell/while/add/y:output:0*
T0*
_output_shapes
: 2
gru/cell/while/addr
gru/cell/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru/cell/while/add_1/y£
gru/cell/while/add_1AddV2*gru_cell_while_gru_cell_while_loop_countergru/cell/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru/cell/while/add_1y
gru/cell/while/IdentityIdentitygru/cell/while/add_1:z:0*
T0*
_output_shapes
: 2
gru/cell/while/Identity
gru/cell/while/Identity_1Identity0gru_cell_while_gru_cell_while_maximum_iterations*
T0*
_output_shapes
: 2
gru/cell/while/Identity_1{
gru/cell/while/Identity_2Identitygru/cell/while/add:z:0*
T0*
_output_shapes
: 2
gru/cell/while/Identity_2¨
gru/cell/while/Identity_3IdentityCgru/cell/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
gru/cell/while/Identity_3
gru/cell/while/Identity_4Identity!gru/cell/while/gru_cell/add_3:z:0*
T0*
_output_shapes
:	2
gru/cell/while/Identity_4"ͺ
Rgru_cell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernelTgru_cell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0"
Fgru_cell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernelHgru_cell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0"
=gru_cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias?gru_cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0"P
%gru_cell_while_gru_cell_strided_slice'gru_cell_while_gru_cell_strided_slice_0";
gru_cell_while_identity gru/cell/while/Identity:output:0"?
gru_cell_while_identity_1"gru/cell/while/Identity_1:output:0"?
gru_cell_while_identity_2"gru/cell/while/Identity_2:output:0"?
gru_cell_while_identity_3"gru/cell/while/Identity_3:output:0"?
gru_cell_while_identity_4"gru/cell/while/Identity_4:output:0"Μ
cgru_cell_while_tensorarrayv2read_tensorlistgetitem_gru_cell_tensorarrayunstack_tensorlistfromtensoregru_cell_while_tensorarrayv2read_tensorlistgetitem_gru_cell_tensorarrayunstack_tensorlistfromtensor_0*6
_input_shapes%
#: : : : :	: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
: 
τP
ξ
>__inference_cell_layer_call_and_return_conditional_losses_3801
inputs_02
.gru_cell_readvariableop_gru_cell_gru_cell_bias;
7gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel6
2gru_cell_matmul_1_readvariableop_gru_cell_variableI
Egru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel
identity’AssignVariableOp’whileu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm}
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*,
_output_shapes
:?????????ΐ2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2β
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape°
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2Ώ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   @  27
5TensorArrayUnstack/TensorListFromTensor/element_shapeψ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2τ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ΐ*
shrink_axis_mask2
strided_slice_1’
gru_cell/ReadVariableOpReadVariableOp.gru_cell_readvariableop_gru_cell_gru_cell_bias*
_output_shapes
:	*
dtype02
gru_cell/ReadVariableOp
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
gru_cell/unstackΊ
gru_cell/MatMul/ReadVariableOpReadVariableOp7gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel* 
_output_shapes
:
ΐ*
dtype02 
gru_cell/MatMul/ReadVariableOp
gru_cell/MatMulMatMulstrided_slice_1:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
gru_cell/MatMul
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*
_output_shapes
:	2
gru_cell/BiasAddb
gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell/Const
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell/split/split_dimΈ
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
gru_cell/splitΈ
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp2gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype02"
 gru_cell/MatMul_1/ReadVariableOpΠ
"gru_cell/MatMul_1/ReadVariableOp_1ReadVariableOpEgru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel* 
_output_shapes
:
*
dtype02$
"gru_cell/MatMul_1/ReadVariableOp_1°
gru_cell/MatMul_1MatMul(gru_cell/MatMul_1/ReadVariableOp:value:0*gru_cell/MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
:	2
gru_cell/MatMul_1
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*
_output_shapes
:	2
gru_cell/BiasAdd_1y
gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ????2
gru_cell/Const_1
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell/split_1/split_dimθ
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const_1:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2
gru_cell/split_1
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*
_output_shapes
:	2
gru_cell/addk
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*
_output_shapes
:	2
gru_cell/Sigmoid
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*
_output_shapes
:	2
gru_cell/add_1q
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*
_output_shapes
:	2
gru_cell/Sigmoid_1
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*
_output_shapes
:	2
gru_cell/mul~
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*
_output_shapes
:	2
gru_cell/add_2d
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*
_output_shapes
:	2
gru_cell/Tanh²
gru_cell/mul_1/ReadVariableOpReadVariableOp2gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype02
gru_cell/mul_1/ReadVariableOp
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0%gru_cell/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
gru_cell/mul_1e
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell/sub/x|
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*
_output_shapes
:	2
gru_cell/subv
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*
_output_shapes
:	2
gru_cell/mul_2{
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*
_output_shapes
:	2
gru_cell/add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
TensorArrayV2_1/element_shapeΆ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
ReadVariableOpReadVariableOp2gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype02
ReadVariableOp
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterΝ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0ReadVariableOp:value:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0.gru_cell_readvariableop_gru_cell_gru_cell_bias7gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernelEgru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*1
_output_shapes
: : : : :	: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_3711*
condR
while_cond_3710*0
output_shapes
: : : : :	: : : : : *
parallel_iterations 2
while΅
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      22
0TensorArrayV2Stack/TensorListStack/element_shapeι
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:?????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¦
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:?????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeσ
AssignVariableOpAssignVariableOp2gru_cell_matmul_1_readvariableop_gru_cell_variablewhile:output:4^ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/mul_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOp
IdentityIdentitystrided_slice_2:output:0^AssignVariableOp^while*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:?????????ΐ::::2$
AssignVariableOpAssignVariableOp2
whilewhile:V R
,
_output_shapes
:?????????ΐ
"
_user_specified_name
inputs/0
ά
·
+__inference_functional_1_layer_call_fn_2562

inputs
stream_conv2d_kernel
stream_conv2d_bias
stream_1_conv2d_1_kernel
stream_1_conv2d_1_bias
gru_cell_gru_cell_bias
gru_cell_gru_cell_kernel
gru_cell_variable&
"gru_cell_gru_cell_recurrent_kernel
dense_kernel

dense_bias
dense_1_kernel
dense_1_bias
dense_2_kernel
dense_2_bias
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstream_conv2d_kernelstream_conv2d_biasstream_1_conv2d_1_kernelstream_1_conv2d_1_biasgru_cell_gru_cell_biasgru_cell_gru_cell_kernelgru_cell_variable"gru_cell_gru_cell_recurrent_kerneldense_kernel
dense_biasdense_1_kerneldense_1_biasdense_2_kerneldense_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*/
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_functional_1_layer_call_and_return_conditional_losses_20532
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:?????????1(::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????1(
 
_user_specified_nameinputs
υ
]
A__inference_reshape_layer_call_and_return_conditional_losses_3084

inputs
identityg
ShapeConst*
_output_shapes
:*
dtype0*%
valueB"   +   $      2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2β
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicem
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape/shape/1e
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :ΐ2
Reshape/shape/2 
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapek
ReshapeReshapeinputsReshape/shape:output:0*
T0*#
_output_shapes
:+ΐ2	
Reshape`
IdentityIdentityReshape:output:0*
T0*#
_output_shapes
:+ΐ2

Identity"
identityIdentity:output:0*%
_input_shapes
:+$:N J
&
_output_shapes
:+$
 
_user_specified_nameinputs
/
ό
F__inference_functional_1_layer_call_and_return_conditional_losses_2053

inputs
stream_stream_conv2d_kernel
stream_stream_conv2d_bias%
!stream_1_stream_1_conv2d_1_kernel#
stream_1_stream_1_conv2d_1_bias
gru_gru_cell_gru_cell_bias 
gru_gru_cell_gru_cell_kernel
gru_gru_cell_variable*
&gru_gru_cell_gru_cell_recurrent_kernel
dense_dense_kernel
dense_dense_bias
dense_1_dense_1_kernel
dense_1_dense_1_bias
dense_2_dense_2_kernel
dense_2_dense_2_bias
identity’dense/StatefulPartitionedCall’dense_1/StatefulPartitionedCall’dense_2/StatefulPartitionedCall’dropout/StatefulPartitionedCall’gru/StatefulPartitionedCall’stream/StatefulPartitionedCall’ stream_1/StatefulPartitionedCallϋ
&tf_op_layer_ExpandDims/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:1(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_14782(
&tf_op_layer_ExpandDims/PartitionedCallΚ
stream/StatefulPartitionedCallStatefulPartitionedCall/tf_op_layer_ExpandDims/PartitionedCall:output:0stream_stream_conv2d_kernelstream_stream_conv2d_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:/&*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_stream_layer_call_and_return_conditional_losses_14972 
stream/StatefulPartitionedCallΤ
 stream_1/StatefulPartitionedCallStatefulPartitionedCall'stream/StatefulPartitionedCall:output:0!stream_1_stream_1_conv2d_1_kernelstream_1_stream_1_conv2d_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:+$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_stream_1_layer_call_and_return_conditional_losses_15202"
 stream_1/StatefulPartitionedCallξ
reshape/PartitionedCallPartitionedCall)stream_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:+ΐ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_15452
reshape/PartitionedCallσ
gru/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0gru_gru_cell_gru_cell_biasgru_gru_cell_gru_cell_kernelgru_gru_cell_variable&gru_gru_cell_gru_cell_recurrent_kernel*
Tin	
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *F
fAR?
=__inference_gru_layer_call_and_return_conditional_losses_18572
gru/StatefulPartitionedCallθ
stream_2/PartitionedCallPartitionedCall$gru/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_stream_2_layer_call_and_return_conditional_losses_18882
stream_2/PartitionedCallϊ
dropout/StatefulPartitionedCallStatefulPartitionedCall!stream_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_19082!
dropout/StatefulPartitionedCall§
dense/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_dense_kerneldense_dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_19362
dense/StatefulPartitionedCall³
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_dense_1_kerneldense_1_dense_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_19592!
dense_1/StatefulPartitionedCall΄
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_dense_2_kerneldense_2_dense_2_bias*
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
GPU 2J 8 *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_19812!
dense_2/StatefulPartitionedCallΫ
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall^gru/StatefulPartitionedCall^stream/StatefulPartitionedCall!^stream_1/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:1(::::::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2@
stream/StatefulPartitionedCallstream/StatefulPartitionedCall2D
 stream_1/StatefulPartitionedCall stream_1/StatefulPartitionedCall:S O
+
_output_shapes
:?????????1(
 
_user_specified_nameinputs
ΛE

cell_while_body_1765&
"cell_while_cell_while_loop_counter,
(cell_while_cell_while_maximum_iterations
cell_while_placeholder
cell_while_placeholder_1
cell_while_placeholder_2#
cell_while_cell_strided_slice_0a
]cell_while_tensorarrayv2read_tensorlistgetitem_cell_tensorarrayunstack_tensorlistfromtensor_0?
;cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0H
Dcell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0T
Pcell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0
cell_while_identity
cell_while_identity_1
cell_while_identity_2
cell_while_identity_3
cell_while_identity_4!
cell_while_cell_strided_slice_
[cell_while_tensorarrayv2read_tensorlistgetitem_cell_tensorarrayunstack_tensorlistfromtensor=
9cell_while_gru_cell_readvariableop_gru_cell_gru_cell_biasF
Bcell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernelR
Ncell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernelΝ
<cell/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   @  2>
<cell/while/TensorArrayV2Read/TensorListGetItem/element_shapeι
.cell/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem]cell_while_tensorarrayv2read_tensorlistgetitem_cell_tensorarrayunstack_tensorlistfromtensor_0cell_while_placeholderEcell/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes
:	ΐ*
element_dtype020
.cell/while/TensorArrayV2Read/TensorListGetItemΕ
"cell/while/gru_cell/ReadVariableOpReadVariableOp;cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0*
_output_shapes
:	*
dtype02$
"cell/while/gru_cell/ReadVariableOp¨
cell/while/gru_cell/unstackUnpack*cell/while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
cell/while/gru_cell/unstackέ
)cell/while/gru_cell/MatMul/ReadVariableOpReadVariableOpDcell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0* 
_output_shapes
:
ΐ*
dtype02+
)cell/while/gru_cell/MatMul/ReadVariableOpΦ
cell/while/gru_cell/MatMulMatMul5cell/while/TensorArrayV2Read/TensorListGetItem:item:01cell/while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/MatMul»
cell/while/gru_cell/BiasAddBiasAdd$cell/while/gru_cell/MatMul:product:0$cell/while/gru_cell/unstack:output:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/BiasAddx
cell/while/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
cell/while/gru_cell/Const
#cell/while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#cell/while/gru_cell/split/split_dimδ
cell/while/gru_cell/splitSplit,cell/while/gru_cell/split/split_dim:output:0$cell/while/gru_cell/BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
cell/while/gru_cell/splitν
+cell/while/gru_cell/MatMul_1/ReadVariableOpReadVariableOpPcell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0* 
_output_shapes
:
*
dtype02-
+cell/while/gru_cell/MatMul_1/ReadVariableOpΏ
cell/while/gru_cell/MatMul_1MatMulcell_while_placeholder_23cell/while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/MatMul_1Α
cell/while/gru_cell/BiasAdd_1BiasAdd&cell/while/gru_cell/MatMul_1:product:0$cell/while/gru_cell/unstack:output:1*
T0*
_output_shapes
:	2
cell/while/gru_cell/BiasAdd_1
cell/while/gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ????2
cell/while/gru_cell/Const_1
%cell/while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%cell/while/gru_cell/split_1/split_dim
cell/while/gru_cell/split_1SplitV&cell/while/gru_cell/BiasAdd_1:output:0$cell/while/gru_cell/Const_1:output:0.cell/while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2
cell/while/gru_cell/split_1―
cell/while/gru_cell/addAddV2"cell/while/gru_cell/split:output:0$cell/while/gru_cell/split_1:output:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/add
cell/while/gru_cell/SigmoidSigmoidcell/while/gru_cell/add:z:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/Sigmoid³
cell/while/gru_cell/add_1AddV2"cell/while/gru_cell/split:output:1$cell/while/gru_cell/split_1:output:1*
T0*
_output_shapes
:	2
cell/while/gru_cell/add_1
cell/while/gru_cell/Sigmoid_1Sigmoidcell/while/gru_cell/add_1:z:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/Sigmoid_1¬
cell/while/gru_cell/mulMul!cell/while/gru_cell/Sigmoid_1:y:0$cell/while/gru_cell/split_1:output:2*
T0*
_output_shapes
:	2
cell/while/gru_cell/mulͺ
cell/while/gru_cell/add_2AddV2"cell/while/gru_cell/split:output:2cell/while/gru_cell/mul:z:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/add_2
cell/while/gru_cell/TanhTanhcell/while/gru_cell/add_2:z:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/Tanh’
cell/while/gru_cell/mul_1Mulcell/while/gru_cell/Sigmoid:y:0cell_while_placeholder_2*
T0*
_output_shapes
:	2
cell/while/gru_cell/mul_1{
cell/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
cell/while/gru_cell/sub/x¨
cell/while/gru_cell/subSub"cell/while/gru_cell/sub/x:output:0cell/while/gru_cell/Sigmoid:y:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/sub’
cell/while/gru_cell/mul_2Mulcell/while/gru_cell/sub:z:0cell/while/gru_cell/Tanh:y:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/mul_2§
cell/while/gru_cell/add_3AddV2cell/while/gru_cell/mul_1:z:0cell/while/gru_cell/mul_2:z:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/add_3υ
/cell/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemcell_while_placeholder_1cell_while_placeholdercell/while/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype021
/cell/while/TensorArrayV2Write/TensorListSetItemf
cell/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
cell/while/add/y}
cell/while/addAddV2cell_while_placeholdercell/while/add/y:output:0*
T0*
_output_shapes
: 2
cell/while/addj
cell/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
cell/while/add_1/y
cell/while/add_1AddV2"cell_while_cell_while_loop_countercell/while/add_1/y:output:0*
T0*
_output_shapes
: 2
cell/while/add_1m
cell/while/IdentityIdentitycell/while/add_1:z:0*
T0*
_output_shapes
: 2
cell/while/Identity
cell/while/Identity_1Identity(cell_while_cell_while_maximum_iterations*
T0*
_output_shapes
: 2
cell/while/Identity_1o
cell/while/Identity_2Identitycell/while/add:z:0*
T0*
_output_shapes
: 2
cell/while/Identity_2
cell/while/Identity_3Identity?cell/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
cell/while/Identity_3
cell/while/Identity_4Identitycell/while/gru_cell/add_3:z:0*
T0*
_output_shapes
:	2
cell/while/Identity_4"@
cell_while_cell_strided_slicecell_while_cell_strided_slice_0"’
Ncell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernelPcell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0"
Bcell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernelDcell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0"x
9cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias;cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0"3
cell_while_identitycell/while/Identity:output:0"7
cell_while_identity_1cell/while/Identity_1:output:0"7
cell_while_identity_2cell/while/Identity_2:output:0"7
cell_while_identity_3cell/while/Identity_3:output:0"7
cell_while_identity_4cell/while/Identity_4:output:0"Ό
[cell_while_tensorarrayv2read_tensorlistgetitem_cell_tensorarrayunstack_tensorlistfromtensor]cell_while_tensorarrayv2read_tensorlistgetitem_cell_tensorarrayunstack_tensorlistfromtensor_0*6
_input_shapes%
#: : : : :	: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
: 


ΰ
'__inference_gru_cell_layer_call_fn_4104

inputs
states_0
gru_cell_gru_cell_bias
gru_cell_gru_cell_kernel&
"gru_cell_gru_cell_recurrent_kernel
identity

identity_1’StatefulPartitionedCallΕ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0gru_cell_gru_cell_biasgru_cell_gru_cell_kernel"gru_cell_gru_cell_recurrent_kernel*
Tin	
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:	:	*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_gru_cell_layer_call_and_return_conditional_losses_10772
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes
:	2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*5
_input_shapes$
":	ΐ:	:::22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	ΐ
 
_user_specified_nameinputs:IE

_output_shapes
:	
"
_user_specified_name
states/0
­=
Ξ
 __inference__traced_restore_4232
file_prefix!
assignvariableop_dense_kernel!
assignvariableop_1_dense_bias%
!assignvariableop_2_dense_1_kernel#
assignvariableop_3_dense_1_bias%
!assignvariableop_4_dense_2_kernel#
assignvariableop_5_dense_2_bias+
'assignvariableop_6_stream_conv2d_kernel)
%assignvariableop_7_stream_conv2d_bias/
+assignvariableop_8_stream_1_conv2d_1_kernel-
)assignvariableop_9_stream_1_conv2d_1_bias0
,assignvariableop_10_gru_cell_gru_cell_kernel:
6assignvariableop_11_gru_cell_gru_cell_recurrent_kernel.
*assignvariableop_12_gru_cell_gru_cell_bias)
%assignvariableop_13_gru_cell_variable
identity_15’AssignVariableOp’AssignVariableOp_1’AssignVariableOp_10’AssignVariableOp_11’AssignVariableOp_12’AssignVariableOp_13’AssignVariableOp_2’AssignVariableOp_3’AssignVariableOp_4’AssignVariableOp_5’AssignVariableOp_6’AssignVariableOp_7’AssignVariableOp_8’AssignVariableOp_9Ρ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*έ
valueΣBΠB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEBFlayer_with_weights-2/gru/keras_api/states/0/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names¬
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesφ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*P
_output_shapes>
<:::::::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1’
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¦
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3€
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¦
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5€
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¬
AssignVariableOp_6AssignVariableOp'assignvariableop_6_stream_conv2d_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7ͺ
AssignVariableOp_7AssignVariableOp%assignvariableop_7_stream_conv2d_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8°
AssignVariableOp_8AssignVariableOp+assignvariableop_8_stream_1_conv2d_1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp)assignvariableop_9_stream_1_conv2d_1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10΄
AssignVariableOp_10AssignVariableOp,assignvariableop_10_gru_cell_gru_cell_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Ύ
AssignVariableOp_11AssignVariableOp6assignvariableop_11_gru_cell_gru_cell_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12²
AssignVariableOp_12AssignVariableOp*assignvariableop_12_gru_cell_gru_cell_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13­
AssignVariableOp_13AssignVariableOp%assignvariableop_13_gru_cell_variableIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_139
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp
Identity_14Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_14
Identity_15IdentityIdentity_14:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_15"#
identity_15Identity_15:output:0*M
_input_shapes<
:: ::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
£K
	
gru_cell_while_body_2429.
*gru_cell_while_gru_cell_while_loop_counter4
0gru_cell_while_gru_cell_while_maximum_iterations
gru_cell_while_placeholder 
gru_cell_while_placeholder_1 
gru_cell_while_placeholder_2+
'gru_cell_while_gru_cell_strided_slice_0i
egru_cell_while_tensorarrayv2read_tensorlistgetitem_gru_cell_tensorarrayunstack_tensorlistfromtensor_0C
?gru_cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0L
Hgru_cell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0X
Tgru_cell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0
gru_cell_while_identity
gru_cell_while_identity_1
gru_cell_while_identity_2
gru_cell_while_identity_3
gru_cell_while_identity_4)
%gru_cell_while_gru_cell_strided_sliceg
cgru_cell_while_tensorarrayv2read_tensorlistgetitem_gru_cell_tensorarrayunstack_tensorlistfromtensorA
=gru_cell_while_gru_cell_readvariableop_gru_cell_gru_cell_biasJ
Fgru_cell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernelV
Rgru_cell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernelΥ
@gru/cell/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   @  2B
@gru/cell/while/TensorArrayV2Read/TensorListGetItem/element_shape
2gru/cell/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemegru_cell_while_tensorarrayv2read_tensorlistgetitem_gru_cell_tensorarrayunstack_tensorlistfromtensor_0gru_cell_while_placeholderIgru/cell/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes
:	ΐ*
element_dtype024
2gru/cell/while/TensorArrayV2Read/TensorListGetItemΡ
&gru/cell/while/gru_cell/ReadVariableOpReadVariableOp?gru_cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0*
_output_shapes
:	*
dtype02(
&gru/cell/while/gru_cell/ReadVariableOp΄
gru/cell/while/gru_cell/unstackUnpack.gru/cell/while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2!
gru/cell/while/gru_cell/unstackι
-gru/cell/while/gru_cell/MatMul/ReadVariableOpReadVariableOpHgru_cell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0* 
_output_shapes
:
ΐ*
dtype02/
-gru/cell/while/gru_cell/MatMul/ReadVariableOpζ
gru/cell/while/gru_cell/MatMulMatMul9gru/cell/while/TensorArrayV2Read/TensorListGetItem:item:05gru/cell/while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2 
gru/cell/while/gru_cell/MatMulΛ
gru/cell/while/gru_cell/BiasAddBiasAdd(gru/cell/while/gru_cell/MatMul:product:0(gru/cell/while/gru_cell/unstack:output:0*
T0*
_output_shapes
:	2!
gru/cell/while/gru_cell/BiasAdd
gru/cell/while/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru/cell/while/gru_cell/Const
'gru/cell/while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'gru/cell/while/gru_cell/split/split_dimτ
gru/cell/while/gru_cell/splitSplit0gru/cell/while/gru_cell/split/split_dim:output:0(gru/cell/while/gru_cell/BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
gru/cell/while/gru_cell/splitω
/gru/cell/while/gru_cell/MatMul_1/ReadVariableOpReadVariableOpTgru_cell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0* 
_output_shapes
:
*
dtype021
/gru/cell/while/gru_cell/MatMul_1/ReadVariableOpΟ
 gru/cell/while/gru_cell/MatMul_1MatMulgru_cell_while_placeholder_27gru/cell/while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2"
 gru/cell/while/gru_cell/MatMul_1Ρ
!gru/cell/while/gru_cell/BiasAdd_1BiasAdd*gru/cell/while/gru_cell/MatMul_1:product:0(gru/cell/while/gru_cell/unstack:output:1*
T0*
_output_shapes
:	2#
!gru/cell/while/gru_cell/BiasAdd_1
gru/cell/while/gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ????2!
gru/cell/while/gru_cell/Const_1‘
)gru/cell/while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2+
)gru/cell/while/gru_cell/split_1/split_dim³
gru/cell/while/gru_cell/split_1SplitV*gru/cell/while/gru_cell/BiasAdd_1:output:0(gru/cell/while/gru_cell/Const_1:output:02gru/cell/while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2!
gru/cell/while/gru_cell/split_1Ώ
gru/cell/while/gru_cell/addAddV2&gru/cell/while/gru_cell/split:output:0(gru/cell/while/gru_cell/split_1:output:0*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/add
gru/cell/while/gru_cell/SigmoidSigmoidgru/cell/while/gru_cell/add:z:0*
T0*
_output_shapes
:	2!
gru/cell/while/gru_cell/SigmoidΓ
gru/cell/while/gru_cell/add_1AddV2&gru/cell/while/gru_cell/split:output:1(gru/cell/while/gru_cell/split_1:output:1*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/add_1
!gru/cell/while/gru_cell/Sigmoid_1Sigmoid!gru/cell/while/gru_cell/add_1:z:0*
T0*
_output_shapes
:	2#
!gru/cell/while/gru_cell/Sigmoid_1Ό
gru/cell/while/gru_cell/mulMul%gru/cell/while/gru_cell/Sigmoid_1:y:0(gru/cell/while/gru_cell/split_1:output:2*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/mulΊ
gru/cell/while/gru_cell/add_2AddV2&gru/cell/while/gru_cell/split:output:2gru/cell/while/gru_cell/mul:z:0*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/add_2
gru/cell/while/gru_cell/TanhTanh!gru/cell/while/gru_cell/add_2:z:0*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/Tanh²
gru/cell/while/gru_cell/mul_1Mul#gru/cell/while/gru_cell/Sigmoid:y:0gru_cell_while_placeholder_2*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/mul_1
gru/cell/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru/cell/while/gru_cell/sub/xΈ
gru/cell/while/gru_cell/subSub&gru/cell/while/gru_cell/sub/x:output:0#gru/cell/while/gru_cell/Sigmoid:y:0*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/sub²
gru/cell/while/gru_cell/mul_2Mulgru/cell/while/gru_cell/sub:z:0 gru/cell/while/gru_cell/Tanh:y:0*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/mul_2·
gru/cell/while/gru_cell/add_3AddV2!gru/cell/while/gru_cell/mul_1:z:0!gru/cell/while/gru_cell/mul_2:z:0*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/add_3
3gru/cell/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_cell_while_placeholder_1gru_cell_while_placeholder!gru/cell/while/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype025
3gru/cell/while/TensorArrayV2Write/TensorListSetItemn
gru/cell/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru/cell/while/add/y
gru/cell/while/addAddV2gru_cell_while_placeholdergru/cell/while/add/y:output:0*
T0*
_output_shapes
: 2
gru/cell/while/addr
gru/cell/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru/cell/while/add_1/y£
gru/cell/while/add_1AddV2*gru_cell_while_gru_cell_while_loop_countergru/cell/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru/cell/while/add_1y
gru/cell/while/IdentityIdentitygru/cell/while/add_1:z:0*
T0*
_output_shapes
: 2
gru/cell/while/Identity
gru/cell/while/Identity_1Identity0gru_cell_while_gru_cell_while_maximum_iterations*
T0*
_output_shapes
: 2
gru/cell/while/Identity_1{
gru/cell/while/Identity_2Identitygru/cell/while/add:z:0*
T0*
_output_shapes
: 2
gru/cell/while/Identity_2¨
gru/cell/while/Identity_3IdentityCgru/cell/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
gru/cell/while/Identity_3
gru/cell/while/Identity_4Identity!gru/cell/while/gru_cell/add_3:z:0*
T0*
_output_shapes
:	2
gru/cell/while/Identity_4"ͺ
Rgru_cell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernelTgru_cell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0"
Fgru_cell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernelHgru_cell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0"
=gru_cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias?gru_cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0"P
%gru_cell_while_gru_cell_strided_slice'gru_cell_while_gru_cell_strided_slice_0";
gru_cell_while_identity gru/cell/while/Identity:output:0"?
gru_cell_while_identity_1"gru/cell/while/Identity_1:output:0"?
gru_cell_while_identity_2"gru/cell/while/Identity_2:output:0"?
gru_cell_while_identity_3"gru/cell/while/Identity_3:output:0"?
gru_cell_while_identity_4"gru/cell/while/Identity_4:output:0"Μ
cgru_cell_while_tensorarrayv2read_tensorlistgetitem_gru_cell_tensorarrayunstack_tensorlistfromtensoregru_cell_while_tensorarrayv2read_tensorlistgetitem_gru_cell_tensorarrayunstack_tensorlistfromtensor_0*6
_input_shapes%
#: : : : :	: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
: 
ϋZ

=__inference_gru_layer_call_and_return_conditional_losses_1857

inputs7
3cell_gru_cell_readvariableop_gru_cell_gru_cell_bias@
<cell_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel;
7cell_gru_cell_matmul_1_readvariableop_gru_cell_variableN
Jcell_gru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel
identity’cell/AssignVariableOp’
cell/while
cell/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
cell/transpose/perm
cell/transpose	Transposeinputscell/transpose/perm:output:0*
T0*#
_output_shapes
:+ΐ2
cell/transposem

cell/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"+      @  2

cell/Shape~
cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
cell/strided_slice/stack
cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cell/strided_slice/stack_1
cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cell/strided_slice/stack_2
cell/strided_sliceStridedSlicecell/Shape:output:0!cell/strided_slice/stack:output:0#cell/strided_slice/stack_1:output:0#cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cell/strided_slice
 cell/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 cell/TensorArrayV2/element_shapeΔ
cell/TensorArrayV2TensorListReserve)cell/TensorArrayV2/element_shape:output:0cell/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
cell/TensorArrayV2Ι
:cell/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   @  2<
:cell/TensorArrayUnstack/TensorListFromTensor/element_shape
,cell/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorcell/transpose:y:0Ccell/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02.
,cell/TensorArrayUnstack/TensorListFromTensor
cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
cell/strided_slice_1/stack
cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cell/strided_slice_1/stack_1
cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cell/strided_slice_1/stack_2
cell/strided_slice_1StridedSlicecell/transpose:y:0#cell/strided_slice_1/stack:output:0%cell/strided_slice_1/stack_1:output:0%cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	ΐ*
shrink_axis_mask2
cell/strided_slice_1±
cell/gru_cell/ReadVariableOpReadVariableOp3cell_gru_cell_readvariableop_gru_cell_gru_cell_bias*
_output_shapes
:	*
dtype02
cell/gru_cell/ReadVariableOp
cell/gru_cell/unstackUnpack$cell/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
cell/gru_cell/unstackΙ
#cell/gru_cell/MatMul/ReadVariableOpReadVariableOp<cell_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel* 
_output_shapes
:
ΐ*
dtype02%
#cell/gru_cell/MatMul/ReadVariableOp¬
cell/gru_cell/MatMulMatMulcell/strided_slice_1:output:0+cell/gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
cell/gru_cell/MatMul£
cell/gru_cell/BiasAddBiasAddcell/gru_cell/MatMul:product:0cell/gru_cell/unstack:output:0*
T0*
_output_shapes
:	2
cell/gru_cell/BiasAddl
cell/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
cell/gru_cell/Const
cell/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
cell/gru_cell/split/split_dimΜ
cell/gru_cell/splitSplit&cell/gru_cell/split/split_dim:output:0cell/gru_cell/BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
cell/gru_cell/splitΗ
%cell/gru_cell/MatMul_1/ReadVariableOpReadVariableOp7cell_gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype02'
%cell/gru_cell/MatMul_1/ReadVariableOpί
'cell/gru_cell/MatMul_1/ReadVariableOp_1ReadVariableOpJcell_gru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel* 
_output_shapes
:
*
dtype02)
'cell/gru_cell/MatMul_1/ReadVariableOp_1Δ
cell/gru_cell/MatMul_1MatMul-cell/gru_cell/MatMul_1/ReadVariableOp:value:0/cell/gru_cell/MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
:	2
cell/gru_cell/MatMul_1©
cell/gru_cell/BiasAdd_1BiasAdd cell/gru_cell/MatMul_1:product:0cell/gru_cell/unstack:output:1*
T0*
_output_shapes
:	2
cell/gru_cell/BiasAdd_1
cell/gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ????2
cell/gru_cell/Const_1
cell/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
cell/gru_cell/split_1/split_dim
cell/gru_cell/split_1SplitV cell/gru_cell/BiasAdd_1:output:0cell/gru_cell/Const_1:output:0(cell/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2
cell/gru_cell/split_1
cell/gru_cell/addAddV2cell/gru_cell/split:output:0cell/gru_cell/split_1:output:0*
T0*
_output_shapes
:	2
cell/gru_cell/addz
cell/gru_cell/SigmoidSigmoidcell/gru_cell/add:z:0*
T0*
_output_shapes
:	2
cell/gru_cell/Sigmoid
cell/gru_cell/add_1AddV2cell/gru_cell/split:output:1cell/gru_cell/split_1:output:1*
T0*
_output_shapes
:	2
cell/gru_cell/add_1
cell/gru_cell/Sigmoid_1Sigmoidcell/gru_cell/add_1:z:0*
T0*
_output_shapes
:	2
cell/gru_cell/Sigmoid_1
cell/gru_cell/mulMulcell/gru_cell/Sigmoid_1:y:0cell/gru_cell/split_1:output:2*
T0*
_output_shapes
:	2
cell/gru_cell/mul
cell/gru_cell/add_2AddV2cell/gru_cell/split:output:2cell/gru_cell/mul:z:0*
T0*
_output_shapes
:	2
cell/gru_cell/add_2s
cell/gru_cell/TanhTanhcell/gru_cell/add_2:z:0*
T0*
_output_shapes
:	2
cell/gru_cell/TanhΑ
"cell/gru_cell/mul_1/ReadVariableOpReadVariableOp7cell_gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype02$
"cell/gru_cell/mul_1/ReadVariableOp’
cell/gru_cell/mul_1Mulcell/gru_cell/Sigmoid:y:0*cell/gru_cell/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
cell/gru_cell/mul_1o
cell/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
cell/gru_cell/sub/x
cell/gru_cell/subSubcell/gru_cell/sub/x:output:0cell/gru_cell/Sigmoid:y:0*
T0*
_output_shapes
:	2
cell/gru_cell/sub
cell/gru_cell/mul_2Mulcell/gru_cell/sub:z:0cell/gru_cell/Tanh:y:0*
T0*
_output_shapes
:	2
cell/gru_cell/mul_2
cell/gru_cell/add_3AddV2cell/gru_cell/mul_1:z:0cell/gru_cell/mul_2:z:0*
T0*
_output_shapes
:	2
cell/gru_cell/add_3
"cell/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2$
"cell/TensorArrayV2_1/element_shapeΚ
cell/TensorArrayV2_1TensorListReserve+cell/TensorArrayV2_1/element_shape:output:0cell/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
cell/TensorArrayV2_1X
	cell/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
	cell/time£
cell/ReadVariableOpReadVariableOp7cell_gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype02
cell/ReadVariableOp
cell/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
cell/while/maximum_iterationst
cell/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
cell/while/loop_counter

cell/whileWhile cell/while/loop_counter:output:0&cell/while/maximum_iterations:output:0cell/time:output:0cell/TensorArrayV2_1:handle:0cell/ReadVariableOp:value:0cell/strided_slice:output:0<cell/TensorArrayUnstack/TensorListFromTensor:output_handle:03cell_gru_cell_readvariableop_gru_cell_gru_cell_bias<cell_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernelJcell_gru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*1
_output_shapes
: : : : :	: : : : : *%
_read_only_resource_inputs
	* 
bodyR
cell_while_body_1765* 
condR
cell_while_cond_1764*0
output_shapes
: : : : :	: : : : : *
parallel_iterations 2

cell/whileΏ
5cell/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      27
5cell/TensorArrayV2Stack/TensorListStack/element_shapeτ
'cell/TensorArrayV2Stack/TensorListStackTensorListStackcell/while:output:3>cell/TensorArrayV2Stack/TensorListStack/element_shape:output:0*#
_output_shapes
:+*
element_dtype02)
'cell/TensorArrayV2Stack/TensorListStack
cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
cell/strided_slice_2/stack
cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
cell/strided_slice_2/stack_1
cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cell/strided_slice_2/stack_2°
cell/strided_slice_2StridedSlice0cell/TensorArrayV2Stack/TensorListStack:tensor:0#cell/strided_slice_2/stack:output:0%cell/strided_slice_2/stack_1:output:0%cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	*
shrink_axis_mask2
cell/strided_slice_2
cell/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
cell/transpose_1/perm±
cell/transpose_1	Transpose0cell/TensorArrayV2Stack/TensorListStack:tensor:0cell/transpose_1/perm:output:0*
T0*#
_output_shapes
:+2
cell/transpose_1p
cell/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
cell/runtime
cell/AssignVariableOpAssignVariableOp7cell_gru_cell_matmul_1_readvariableop_gru_cell_variablecell/while:output:4^cell/ReadVariableOp&^cell/gru_cell/MatMul_1/ReadVariableOp#^cell/gru_cell/mul_1/ReadVariableOp*
_output_shapes
 *
dtype02
cell/AssignVariableOpb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimscell/strided_slice_2:output:0ExpandDims/dim:output:0*
T0*#
_output_shapes
:2

ExpandDims
IdentityIdentityExpandDims:output:0^cell/AssignVariableOp^cell/while*
T0*#
_output_shapes
:2

Identity"
identityIdentity:output:0*2
_input_shapes!
:+ΐ::::2.
cell/AssignVariableOpcell/AssignVariableOp2

cell/while
cell/while:K G
#
_output_shapes
:+ΐ
 
_user_specified_nameinputs
Ϋ
l
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_1478

inputs
identityk
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims/dim

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*
_cloned(*&
_output_shapes
:1(2

ExpandDimsf
IdentityIdentityExpandDims:output:0*
T0*&
_output_shapes
:1(2

Identity"
identityIdentity:output:0*!
_input_shapes
:1(:J F
"
_output_shapes
:1(
 
_user_specified_nameinputs

³
A__inference_dense_1_layer_call_and_return_conditional_losses_3477

inputs(
$matmul_readvariableop_dense_1_kernel'
#biasadd_readvariableop_dense_1_bias
identity
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_1_kernel* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpk
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
MatMul
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_1_bias*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2	
BiasAddP
ReluReluBiasAdd:output:0*
T0*
_output_shapes
:	2
Relu^
IdentityIdentityRelu:activations:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*&
_input_shapes
:	:::G C

_output_shapes
:	
 
_user_specified_nameinputs
Ι

&__inference_dense_2_layer_call_fn_3501

inputs
dense_2_kernel
dense_2_bias
identity’StatefulPartitionedCallς
StatefulPartitionedCallStatefulPartitionedCallinputsdense_2_kerneldense_2_bias*
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
GPU 2J 8 *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_19812
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*&
_input_shapes
:	::22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	
 
_user_specified_nameinputs
Λ

&__inference_dense_1_layer_call_fn_3484

inputs
dense_1_kernel
dense_1_bias
identity’StatefulPartitionedCallσ
StatefulPartitionedCallStatefulPartitionedCallinputsdense_1_kerneldense_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_19592
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*&
_input_shapes
:	::22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	
 
_user_specified_nameinputs"ΈL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*
serving_default
6
input_1+
serving_default_input_1:01(2
dense_2'
StatefulPartitionedCall:0tensorflow/serving/predict:
[
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
	variables
regularization_losses
trainable_variables
	keras_api

signatures
+΅&call_and_return_all_conditional_losses
Ά_default_save_signature
·__call__"°W
_tf_keras_networkW{"class_name": "Functional", "name": "functional_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 49, 40]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "ExpandDims", "trainable": true, "dtype": "float32", "node_def": {"name": "ExpandDims", "op": "ExpandDims", "input": ["input_1", "ExpandDims/dim"], "attr": {"Tdim": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": -1}}, "name": "tf_op_layer_ExpandDims", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Stream", "config": {"name": "stream", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "TRAINING", "pad_time_dim": false, "cell": {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "__passive_serialization__": true}, "state_shape": [1, 3, 40, 1], "ring_buffer_size_in_time_dim": 3}, "name": "stream", "inbound_nodes": [[["tf_op_layer_ExpandDims", 0, 0, {}]]]}, {"class_name": "Stream", "config": {"name": "stream_1", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "TRAINING", "pad_time_dim": false, "cell": {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "__passive_serialization__": true}, "state_shape": [1, 5, 38, 16], "ring_buffer_size_in_time_dim": 5}, "name": "stream_1", "inbound_nodes": [[["stream", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1, 576]}}, "name": "reshape", "inbound_nodes": [[["stream_1", 0, 0, {}]]]}, {"class_name": "GRU", "config": {"name": "gru", "trainable": true, "dtype": "float32", "mode": "TRAINING", "inference_batch_size": 1, "units": 256, "return_sequences": 0, "unroll": false, "stateful": true}, "name": "gru", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "Stream", "config": {"name": "stream_2", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "TRAINING", "pad_time_dim": false, "cell": {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "__passive_serialization__": true}, "state_shape": [1, 1, 256], "ring_buffer_size_in_time_dim": null}, "name": "stream_2", "inbound_nodes": [[["gru", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["stream_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 14, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense_2", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 49, 40]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 49, 40]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "ExpandDims", "trainable": true, "dtype": "float32", "node_def": {"name": "ExpandDims", "op": "ExpandDims", "input": ["input_1", "ExpandDims/dim"], "attr": {"Tdim": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": -1}}, "name": "tf_op_layer_ExpandDims", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Stream", "config": {"name": "stream", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "TRAINING", "pad_time_dim": false, "cell": {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "__passive_serialization__": true}, "state_shape": [1, 3, 40, 1], "ring_buffer_size_in_time_dim": 3}, "name": "stream", "inbound_nodes": [[["tf_op_layer_ExpandDims", 0, 0, {}]]]}, {"class_name": "Stream", "config": {"name": "stream_1", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "TRAINING", "pad_time_dim": false, "cell": {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "__passive_serialization__": true}, "state_shape": [1, 5, 38, 16], "ring_buffer_size_in_time_dim": 5}, "name": "stream_1", "inbound_nodes": [[["stream", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1, 576]}}, "name": "reshape", "inbound_nodes": [[["stream_1", 0, 0, {}]]]}, {"class_name": "GRU", "config": {"name": "gru", "trainable": true, "dtype": "float32", "mode": "TRAINING", "inference_batch_size": 1, "units": 256, "return_sequences": 0, "unroll": false, "stateful": true}, "name": "gru", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "Stream", "config": {"name": "stream_2", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "TRAINING", "pad_time_dim": false, "cell": {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "__passive_serialization__": true}, "state_shape": [1, 1, 256], "ring_buffer_size_in_time_dim": null}, "name": "stream_2", "inbound_nodes": [[["gru", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["stream_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 14, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense_2", 0, 0]]}}}
ν"κ
_tf_keras_input_layerΚ{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [1, 49, 40]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 49, 40]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}

	variables
regularization_losses
trainable_variables
	keras_api
+Έ&call_and_return_all_conditional_losses
Ή__call__"ϊ
_tf_keras_layerΰ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_ExpandDims", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "ExpandDims", "trainable": true, "dtype": "float32", "node_def": {"name": "ExpandDims", "op": "ExpandDims", "input": ["input_1", "ExpandDims/dim"], "attr": {"Tdim": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": -1}}}
λ

cell
state_shape
	variables
regularization_losses
trainable_variables
	keras_api
+Ί&call_and_return_all_conditional_losses
»__call__"Ώ	
_tf_keras_layer₯	{"class_name": "Stream", "name": "stream", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "stream", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "TRAINING", "pad_time_dim": false, "cell": {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "__passive_serialization__": true}, "state_shape": [1, 3, 40, 1], "ring_buffer_size_in_time_dim": 3}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 49, 40, 1]}}
σ

cell
state_shape
	variables
regularization_losses
trainable_variables
 	keras_api
+Ό&call_and_return_all_conditional_losses
½__call__"Η	
_tf_keras_layer­	{"class_name": "Stream", "name": "stream_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "stream_1", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "TRAINING", "pad_time_dim": false, "cell": {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "__passive_serialization__": true}, "state_shape": [1, 5, 38, 16], "ring_buffer_size_in_time_dim": 5}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 47, 38, 16]}}
υ
!	variables
"regularization_losses
#trainable_variables
$	keras_api
+Ύ&call_and_return_all_conditional_losses
Ώ__call__"δ
_tf_keras_layerΚ{"class_name": "Reshape", "name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1, 576]}}}
§
%gru
&	variables
'regularization_losses
(trainable_variables
)	keras_api
+ΐ&call_and_return_all_conditional_losses
Α__call__"
_tf_keras_layerσ{"class_name": "GRU", "name": "gru", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": true, "must_restore_from_config": false, "config": {"name": "gru", "trainable": true, "dtype": "float32", "mode": "TRAINING", "inference_batch_size": 1, "units": 256, "return_sequences": 0, "unroll": false, "stateful": true}}
Ξ
*cell
+state_shape
,	variables
-regularization_losses
.trainable_variables
/	keras_api
+Β&call_and_return_all_conditional_losses
Γ__call__"’
_tf_keras_layer{"class_name": "Stream", "name": "stream_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "stream_2", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "TRAINING", "pad_time_dim": false, "cell": {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "__passive_serialization__": true}, "state_shape": [1, 1, 256], "ring_buffer_size_in_time_dim": null}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 1, 256]}}
γ
0	variables
1regularization_losses
2trainable_variables
3	keras_api
+Δ&call_and_return_all_conditional_losses
Ε__call__"?
_tf_keras_layerΈ{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
©

4kernel
5bias
6	variables
7regularization_losses
8trainable_variables
9	keras_api
+Ζ&call_and_return_all_conditional_losses
Η__call__"
_tf_keras_layerθ{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}}
«

:kernel
;bias
<	variables
=regularization_losses
>trainable_variables
?	keras_api
+Θ&call_and_return_all_conditional_losses
Ι__call__"
_tf_keras_layerκ{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}
¬

@kernel
Abias
B	variables
Cregularization_losses
Dtrainable_variables
E	keras_api
+Κ&call_and_return_all_conditional_losses
Λ__call__"
_tf_keras_layerλ{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 14, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}}
~
F0
G1
H2
I3
J4
K5
L6
47
58
:9
;10
@11
A12"
trackable_list_wrapper
 "
trackable_list_wrapper
~
F0
G1
H2
I3
J4
K5
L6
47
58
:9
;10
@11
A12"
trackable_list_wrapper
Ξ
Mlayer_metrics

Nlayers
	variables
Ometrics
Pnon_trainable_variables
regularization_losses
Qlayer_regularization_losses
trainable_variables
·__call__
Ά_default_save_signature
+΅&call_and_return_all_conditional_losses
'΅"call_and_return_conditional_losses"
_generic_user_object
-
Μserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
Rlayer_metrics

Slayers
	variables
Tmetrics
Unon_trainable_variables
regularization_losses
Vlayer_regularization_losses
trainable_variables
Ή__call__
+Έ&call_and_return_all_conditional_losses
'Έ"call_and_return_conditional_losses"
_generic_user_object
	

Fkernel
Gbias
W	variables
Xregularization_losses
Ytrainable_variables
Z	keras_api
+Ν&call_and_return_all_conditional_losses
Ξ__call__"ψ
_tf_keras_layerή{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}}
 "
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
°
[layer_metrics

\layers
	variables
]metrics
^non_trainable_variables
regularization_losses
_layer_regularization_losses
trainable_variables
»__call__
+Ί&call_and_return_all_conditional_losses
'Ί"call_and_return_conditional_losses"
_generic_user_object
€	

Hkernel
Ibias
`	variables
aregularization_losses
btrainable_variables
c	keras_api
+Ο&call_and_return_all_conditional_losses
Π__call__"ύ
_tf_keras_layerγ{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}}
 "
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
°
dlayer_metrics

elayers
	variables
fmetrics
gnon_trainable_variables
regularization_losses
hlayer_regularization_losses
trainable_variables
½__call__
+Ό&call_and_return_all_conditional_losses
'Ό"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
ilayer_metrics

jlayers
!	variables
kmetrics
lnon_trainable_variables
"regularization_losses
mlayer_regularization_losses
#trainable_variables
Ώ__call__
+Ύ&call_and_return_all_conditional_losses
'Ύ"call_and_return_conditional_losses"
_generic_user_object
α

ncell
o
state_spec
p	variables
qregularization_losses
rtrainable_variables
s	keras_api
+Ρ&call_and_return_all_conditional_losses
?__call__"Ά	
_tf_keras_rnn_layer	{"class_name": "GRU", "name": "cell", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": true, "must_restore_from_config": false, "config": {"name": "cell", "trainable": true, "dtype": "float32", "return_sequences": 0, "return_state": false, "go_backwards": false, "stateful": true, "unroll": false, "time_major": false, "units": 256, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [1, null, 576]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}]}
5
J0
K1
L2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
J0
K1
L2"
trackable_list_wrapper
°
tlayer_metrics

ulayers
&	variables
vmetrics
wnon_trainable_variables
'regularization_losses
xlayer_regularization_losses
(trainable_variables
Α__call__
+ΐ&call_and_return_all_conditional_losses
'ΐ"call_and_return_conditional_losses"
_generic_user_object
δ
y	variables
zregularization_losses
{trainable_variables
|	keras_api
+Σ&call_and_return_all_conditional_losses
Τ__call__"Σ
_tf_keras_layerΉ{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
}layer_metrics

~layers
,	variables
metrics
non_trainable_variables
-regularization_losses
 layer_regularization_losses
.trainable_variables
Γ__call__
+Β&call_and_return_all_conditional_losses
'Β"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
layer_metrics
layers
0	variables
metrics
non_trainable_variables
1regularization_losses
 layer_regularization_losses
2trainable_variables
Ε__call__
+Δ&call_and_return_all_conditional_losses
'Δ"call_and_return_conditional_losses"
_generic_user_object
 :
2dense/kernel
:2
dense/bias
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
΅
layer_metrics
layers
6	variables
metrics
non_trainable_variables
7regularization_losses
 layer_regularization_losses
8trainable_variables
Η__call__
+Ζ&call_and_return_all_conditional_losses
'Ζ"call_and_return_conditional_losses"
_generic_user_object
": 
2dense_1/kernel
:2dense_1/bias
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
΅
layer_metrics
layers
<	variables
metrics
non_trainable_variables
=regularization_losses
 layer_regularization_losses
>trainable_variables
Ι__call__
+Θ&call_and_return_all_conditional_losses
'Θ"call_and_return_conditional_losses"
_generic_user_object
!:	2dense_2/kernel
:2dense_2/bias
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
΅
layer_metrics
layers
B	variables
metrics
non_trainable_variables
Cregularization_losses
 layer_regularization_losses
Dtrainable_variables
Λ__call__
+Κ&call_and_return_all_conditional_losses
'Κ"call_and_return_conditional_losses"
_generic_user_object
.:,2stream/conv2d/kernel
 :2stream/conv2d/bias
2:02stream_1/conv2d_1/kernel
$:"2stream_1/conv2d_1/bias
,:*
ΐ2gru/cell/gru_cell/kernel
6:4
2"gru/cell/gru_cell/recurrent_kernel
):'	2gru/cell/gru_cell/bias
 "
trackable_dict_wrapper
n
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
10"
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
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
΅
layer_metrics
layers
W	variables
metrics
non_trainable_variables
Xregularization_losses
 layer_regularization_losses
Ytrainable_variables
Ξ__call__
+Ν&call_and_return_all_conditional_losses
'Ν"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
΅
layer_metrics
layers
`	variables
metrics
non_trainable_variables
aregularization_losses
 layer_regularization_losses
btrainable_variables
Π__call__
+Ο&call_and_return_all_conditional_losses
'Ο"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
'
0"
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
£

Jkernel
Krecurrent_kernel
Lbias
 	variables
‘regularization_losses
’trainable_variables
£	keras_api
+Υ&call_and_return_all_conditional_losses
Φ__call__"β
_tf_keras_layerΘ{"class_name": "GRUCell", "name": "gru_cell", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gru_cell", "trainable": true, "dtype": "float32", "units": 256, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}
 "
trackable_list_wrapper
5
J0
K1
L2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
J0
K1
L2"
trackable_list_wrapper
Β
€layer_metrics
₯layers
p	variables
¦states
§metrics
¨non_trainable_variables
qregularization_losses
 ©layer_regularization_losses
rtrainable_variables
?__call__
+Ρ&call_and_return_all_conditional_losses
'Ρ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
'
%0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
ͺlayer_metrics
«layers
y	variables
¬metrics
­non_trainable_variables
zregularization_losses
 ?layer_regularization_losses
{trainable_variables
Τ__call__
+Σ&call_and_return_all_conditional_losses
'Σ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
'
*0"
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
5
J0
K1
L2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
J0
K1
L2"
trackable_list_wrapper
Έ
―layer_metrics
°layers
 	variables
±metrics
²non_trainable_variables
‘regularization_losses
 ³layer_regularization_losses
’trainable_variables
Φ__call__
+Υ&call_and_return_all_conditional_losses
'Υ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
'
n0"
trackable_list_wrapper
(
΄0"
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
$:"	2gru/cell/Variable
ζ2γ
F__inference_functional_1_layer_call_and_return_conditional_losses_2344
F__inference_functional_1_layer_call_and_return_conditional_losses_2986
F__inference_functional_1_layer_call_and_return_conditional_losses_2787
F__inference_functional_1_layer_call_and_return_conditional_losses_2543ΐ
·²³
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
kwonlydefaultsͺ 
annotationsͺ *
 
ΰ2έ
__inference__wrapped_model_877Ί
²
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
annotationsͺ **’'
%"
input_1?????????1(
ϊ2χ
+__inference_functional_1_layer_call_fn_3005
+__inference_functional_1_layer_call_fn_3024
+__inference_functional_1_layer_call_fn_2562
+__inference_functional_1_layer_call_fn_2581ΐ
·²³
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
kwonlydefaultsͺ 
annotationsͺ *
 
ϊ2χ
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_3030’
²
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
annotationsͺ *
 
ί2ά
5__inference_tf_op_layer_ExpandDims_layer_call_fn_3035’
²
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
annotationsͺ *
 
κ2η
@__inference_stream_layer_call_and_return_conditional_losses_3046’
²
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
annotationsͺ *
 
Ο2Μ
%__inference_stream_layer_call_fn_3053’
²
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
annotationsͺ *
 
μ2ι
B__inference_stream_1_layer_call_and_return_conditional_losses_3064’
²
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
annotationsͺ *
 
Ρ2Ξ
'__inference_stream_1_layer_call_fn_3071’
²
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
annotationsͺ *
 
λ2θ
A__inference_reshape_layer_call_and_return_conditional_losses_3084’
²
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
annotationsͺ *
 
Π2Ν
&__inference_reshape_layer_call_fn_3089’
²
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
annotationsͺ *
 
·2΄
=__inference_gru_layer_call_and_return_conditional_losses_3393
=__inference_gru_layer_call_and_return_conditional_losses_3241³
ͺ²¦
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

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
2ώ
"__inference_gru_layer_call_fn_3411
"__inference_gru_layer_call_fn_3402³
ͺ²¦
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

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
μ2ι
B__inference_stream_2_layer_call_and_return_conditional_losses_3417’
²
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
annotationsͺ *
 
Ρ2Ξ
'__inference_stream_2_layer_call_fn_3422’
²
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
annotationsͺ *
 
ΐ2½
A__inference_dropout_layer_call_and_return_conditional_losses_3439
A__inference_dropout_layer_call_and_return_conditional_losses_3434΄
«²§
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
kwonlydefaultsͺ 
annotationsͺ *
 
2
&__inference_dropout_layer_call_fn_3444
&__inference_dropout_layer_call_fn_3449΄
«²§
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
kwonlydefaultsͺ 
annotationsͺ *
 
ι2ζ
?__inference_dense_layer_call_and_return_conditional_losses_3459’
²
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
annotationsͺ *
 
Ξ2Λ
$__inference_dense_layer_call_fn_3466’
²
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
annotationsͺ *
 
λ2θ
A__inference_dense_1_layer_call_and_return_conditional_losses_3477’
²
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
annotationsͺ *
 
Π2Ν
&__inference_dense_1_layer_call_fn_3484’
²
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
annotationsͺ *
 
λ2θ
A__inference_dense_2_layer_call_and_return_conditional_losses_3494’
²
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
annotationsͺ *
 
Π2Ν
&__inference_dense_2_layer_call_fn_3501’
²
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
annotationsͺ *
 
1B/
"__inference_signature_wrapper_2138input_1
¨2₯’
²
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
annotationsͺ *
 
¨2₯’
²
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
annotationsͺ *
 
¨2₯’
²
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
annotationsͺ *
 
¨2₯’
²
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
annotationsͺ *
 
Ϋ2Ψ
>__inference_cell_layer_call_and_return_conditional_losses_3801
>__inference_cell_layer_call_and_return_conditional_losses_3651Υ
Μ²Θ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
₯2’
#__inference_cell_layer_call_fn_3810
#__inference_cell_layer_call_fn_3819Υ
Μ²Θ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
¨2₯’
²
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
annotationsͺ *
 
¨2₯’
²
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
annotationsͺ *
 
Τ2Ρ
B__inference_gru_cell_layer_call_and_return_conditional_losses_3862
B__inference_gru_cell_layer_call_and_return_conditional_losses_3905
B__inference_gru_cell_layer_call_and_return_conditional_losses_4093
B__inference_gru_cell_layer_call_and_return_conditional_losses_4053Ύ
΅²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
θ2ε
'__inference_gru_cell_layer_call_fn_4013
'__inference_gru_cell_layer_call_fn_4104
'__inference_gru_cell_layer_call_fn_4115
'__inference_gru_cell_layer_call_fn_3959Ύ
΅²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
__inference__wrapped_model_877qFGHILJ΄K45:;@A4’1
*’'
%"
input_1?????????1(
ͺ "(ͺ%
#
dense_2
dense_2±
>__inference_cell_layer_call_and_return_conditional_losses_3651oLJ΄KG’D
=’:
,)
'$
inputs/0?????????ΐ

 
p

 
ͺ "’

0	
 ±
>__inference_cell_layer_call_and_return_conditional_losses_3801oLJ΄KG’D
=’:
,)
'$
inputs/0?????????ΐ

 
p 

 
ͺ "’

0	
 
#__inference_cell_layer_call_fn_3810b΄LJKG’D
=’:
,)
'$
inputs/0?????????ΐ

 
p

 
ͺ "	
#__inference_cell_layer_call_fn_3819b΄LJKG’D
=’:
,)
'$
inputs/0?????????ΐ

 
p 

 
ͺ "	
A__inference_dense_1_layer_call_and_return_conditional_losses_3477L:;'’$
’

inputs	
ͺ "’

0	
 i
&__inference_dense_1_layer_call_fn_3484?:;'’$
’

inputs	
ͺ "	
A__inference_dense_2_layer_call_and_return_conditional_losses_3494K@A'’$
’

inputs	
ͺ "’

0
 h
&__inference_dense_2_layer_call_fn_3501>@A'’$
’

inputs	
ͺ "
?__inference_dense_layer_call_and_return_conditional_losses_3459L45'’$
’

inputs	
ͺ "’

0	
 g
$__inference_dense_layer_call_fn_3466?45'’$
’

inputs	
ͺ "	
A__inference_dropout_layer_call_and_return_conditional_losses_3434L+’(
!’

inputs	
p
ͺ "’

0	
 
A__inference_dropout_layer_call_and_return_conditional_losses_3439L+’(
!’

inputs	
p 
ͺ "’

0	
 i
&__inference_dropout_layer_call_fn_3444?+’(
!’

inputs	
p
ͺ "	i
&__inference_dropout_layer_call_fn_3449?+’(
!’

inputs	
p 
ͺ "	Ά
F__inference_functional_1_layer_call_and_return_conditional_losses_2344lFGHILJ΄K45:;@A;’8
1’.
$!
inputs?????????1(
p

 
ͺ "’

0
 Ά
F__inference_functional_1_layer_call_and_return_conditional_losses_2543lFGHILJ΄K45:;@A;’8
1’.
$!
inputs?????????1(
p 

 
ͺ "’

0
 ·
F__inference_functional_1_layer_call_and_return_conditional_losses_2787mFGHILJ΄K45:;@A<’9
2’/
%"
input_1?????????1(
p

 
ͺ "’

0
 ·
F__inference_functional_1_layer_call_and_return_conditional_losses_2986mFGHILJ΄K45:;@A<’9
2’/
%"
input_1?????????1(
p 

 
ͺ "’

0
 
+__inference_functional_1_layer_call_fn_2562_FGHILJ΄K45:;@A;’8
1’.
$!
inputs?????????1(
p

 
ͺ "
+__inference_functional_1_layer_call_fn_2581_FGHILJ΄K45:;@A;’8
1’.
$!
inputs?????????1(
p 

 
ͺ "
+__inference_functional_1_layer_call_fn_3005`FGHILJ΄K45:;@A<’9
2’/
%"
input_1?????????1(
p

 
ͺ "
+__inference_functional_1_layer_call_fn_3024`FGHILJ΄K45:;@A<’9
2’/
%"
input_1?????????1(
p 

 
ͺ "ν
B__inference_gru_cell_layer_call_and_return_conditional_losses_3862¦LJKi’f
_’\

inputs	ΐ
<’9
74	"’
ϊ	


jstates/0VariableSpec
p
ͺ "4’1
*’'

0/0


0/1/0
 ν
B__inference_gru_cell_layer_call_and_return_conditional_losses_3905¦LJKi’f
_’\

inputs	ΐ
<’9
74	"’
ϊ	


jstates/0VariableSpec
p 
ͺ "4’1
*’'

0/0


0/1/0
 ή
B__inference_gru_cell_layer_call_and_return_conditional_losses_4053LJKL’I
B’?

inputs	ΐ
’

states/0	
p
ͺ "B’?
8’5

0/0	


0/1/0	
 ή
B__inference_gru_cell_layer_call_and_return_conditional_losses_4093LJKL’I
B’?

inputs	ΐ
’

states/0	
p 
ͺ "B’?
8’5

0/0	


0/1/0	
 Δ
'__inference_gru_cell_layer_call_fn_3959LJKi’f
_’\

inputs	ΐ
<’9
74	"’
ϊ	


jstates/0VariableSpec
p
ͺ "&’#
	
0


1/0Δ
'__inference_gru_cell_layer_call_fn_4013LJKi’f
_’\

inputs	ΐ
<’9
74	"’
ϊ	


jstates/0VariableSpec
p 
ͺ "&’#
	
0


1/0΅
'__inference_gru_cell_layer_call_fn_4104LJKL’I
B’?

inputs	ΐ
’

states/0	
p
ͺ "4’1

0	


1/0	΅
'__inference_gru_cell_layer_call_fn_4115LJKL’I
B’?

inputs	ΐ
’

states/0	
p 
ͺ "4’1

0	


1/0	
=__inference_gru_layer_call_and_return_conditional_losses_3241[LJ΄K/’,
%’"

inputs+ΐ
p
ͺ "!’

0
 
=__inference_gru_layer_call_and_return_conditional_losses_3393[LJ΄K/’,
%’"

inputs+ΐ
p 
ͺ "!’

0
 t
"__inference_gru_layer_call_fn_3402NLJ΄K/’,
%’"

inputs+ΐ
p
ͺ "t
"__inference_gru_layer_call_fn_3411NLJ΄K/’,
%’"

inputs+ΐ
p 
ͺ "
A__inference_reshape_layer_call_and_return_conditional_losses_3084S.’+
$’!

inputs+$
ͺ "!’

0+ΐ
 p
&__inference_reshape_layer_call_fn_3089F.’+
$’!

inputs+$
ͺ "+ΐ
"__inference_signature_wrapper_2138sFGHILJ΄K45:;@A6’3
’ 
,ͺ)
'
input_1
input_11("(ͺ%
#
dense_2
dense_2 
B__inference_stream_1_layer_call_and_return_conditional_losses_3064ZHI.’+
$’!

inputs/&
ͺ "$’!

0+$
 x
'__inference_stream_1_layer_call_fn_3071MHI.’+
$’!

inputs/&
ͺ "+$
B__inference_stream_2_layer_call_and_return_conditional_losses_3417L+’(
!’

inputs
ͺ "’

0	
 j
'__inference_stream_2_layer_call_fn_3422?+’(
!’

inputs
ͺ "	
@__inference_stream_layer_call_and_return_conditional_losses_3046ZFG.’+
$’!

inputs1(
ͺ "$’!

0/&
 v
%__inference_stream_layer_call_fn_3053MFG.’+
$’!

inputs1(
ͺ "/&¦
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_3030R*’'
 ’

inputs1(
ͺ "$’!

01(
 ~
5__inference_tf_op_layer_ExpandDims_layer_call_fn_3035E*’'
 ’

inputs1(
ͺ "1(