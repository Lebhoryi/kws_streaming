µÁ
Ý£
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
¾
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
 "serve*2.3.0-dev202005152v1.12.1-31980-g2b2e4412058ãÈ

streaming/stream/statesVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namestreaming/stream/states

+streaming/stream/states/Read/ReadVariableOpReadVariableOpstreaming/stream/states*&
_output_shapes
:*
dtype0

streaming/stream_1/statesVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_namestreaming/stream_1/states

-streaming/stream_1/states/Read/ReadVariableOpReadVariableOpstreaming/stream_1/states*&
_output_shapes
:@*
dtype0

streaming/stream_2/statesVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_namestreaming/stream_2/states

-streaming/stream_2/states/Read/ReadVariableOpReadVariableOpstreaming/stream_2/states*&
_output_shapes
:@*
dtype0

streaming/stream_3/statesVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_namestreaming/stream_3/states

-streaming/stream_3/states/Read/ReadVariableOpReadVariableOpstreaming/stream_3/states*&
_output_shapes
:@*
dtype0

streaming/stream_4/statesVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_namestreaming/stream_4/states

-streaming/stream_4/states/Read/ReadVariableOpReadVariableOpstreaming/stream_4/states*&
_output_shapes
:@*
dtype0

streaming/stream_5/statesVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namestreaming/stream_5/states

-streaming/stream_5/states/Read/ReadVariableOpReadVariableOpstreaming/stream_5/states*'
_output_shapes
:*
dtype0

streaming/stream_6/statesVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_namestreaming/stream_6/states

-streaming/stream_6/states/Read/ReadVariableOpReadVariableOpstreaming/stream_6/states*&
_output_shapes
:@*
dtype0

streaming/stream_7/statesVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namestreaming/stream_7/states

-streaming/stream_7/states/Read/ReadVariableOpReadVariableOpstreaming/stream_7/states*'
_output_shapes
:*
dtype0

streaming/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_namestreaming/dense/kernel

*streaming/dense/kernel/Read/ReadVariableOpReadVariableOpstreaming/dense/kernel* 
_output_shapes
:
*
dtype0

streaming/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_namestreaming/dense/bias
z
(streaming/dense/bias/Read/ReadVariableOpReadVariableOpstreaming/dense/bias*
_output_shapes	
:*
dtype0

streaming/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*)
shared_namestreaming/dense_1/kernel

,streaming/dense_1/kernel/Read/ReadVariableOpReadVariableOpstreaming/dense_1/kernel* 
_output_shapes
:
*
dtype0

streaming/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namestreaming/dense_1/bias
~
*streaming/dense_1/bias/Read/ReadVariableOpReadVariableOpstreaming/dense_1/bias*
_output_shapes	
:*
dtype0

streaming/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*)
shared_namestreaming/dense_2/kernel

,streaming/dense_2/kernel/Read/ReadVariableOpReadVariableOpstreaming/dense_2/kernel*
_output_shapes
:	*
dtype0

streaming/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namestreaming/dense_2/bias
}
*streaming/dense_2/bias/Read/ReadVariableOpReadVariableOpstreaming/dense_2/bias*
_output_shapes
:*
dtype0

stream/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_namestream/conv2d/kernel

(stream/conv2d/kernel/Read/ReadVariableOpReadVariableOpstream/conv2d/kernel*&
_output_shapes
:@*
dtype0
|
stream/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_namestream/conv2d/bias
u
&stream/conv2d/bias/Read/ReadVariableOpReadVariableOpstream/conv2d/bias*
_output_shapes
:@*
dtype0

stream_1/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*)
shared_namestream_1/conv2d_1/kernel

,stream_1/conv2d_1/kernel/Read/ReadVariableOpReadVariableOpstream_1/conv2d_1/kernel*&
_output_shapes
:@@*
dtype0

stream_1/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_namestream_1/conv2d_1/bias
}
*stream_1/conv2d_1/bias/Read/ReadVariableOpReadVariableOpstream_1/conv2d_1/bias*
_output_shapes
:@*
dtype0

stream_2/conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*)
shared_namestream_2/conv2d_2/kernel

,stream_2/conv2d_2/kernel/Read/ReadVariableOpReadVariableOpstream_2/conv2d_2/kernel*&
_output_shapes
:@@*
dtype0

stream_2/conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_namestream_2/conv2d_2/bias
}
*stream_2/conv2d_2/bias/Read/ReadVariableOpReadVariableOpstream_2/conv2d_2/bias*
_output_shapes
:@*
dtype0

stream_3/conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*)
shared_namestream_3/conv2d_3/kernel

,stream_3/conv2d_3/kernel/Read/ReadVariableOpReadVariableOpstream_3/conv2d_3/kernel*&
_output_shapes
:@@*
dtype0

stream_3/conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_namestream_3/conv2d_3/bias
}
*stream_3/conv2d_3/bias/Read/ReadVariableOpReadVariableOpstream_3/conv2d_3/bias*
_output_shapes
:@*
dtype0

stream_4/conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_namestream_4/conv2d_4/kernel

,stream_4/conv2d_4/kernel/Read/ReadVariableOpReadVariableOpstream_4/conv2d_4/kernel*'
_output_shapes
:@*
dtype0

stream_4/conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namestream_4/conv2d_4/bias
~
*stream_4/conv2d_4/bias/Read/ReadVariableOpReadVariableOpstream_4/conv2d_4/bias*
_output_shapes	
:*
dtype0

stream_5/conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_namestream_5/conv2d_5/kernel

,stream_5/conv2d_5/kernel/Read/ReadVariableOpReadVariableOpstream_5/conv2d_5/kernel*'
_output_shapes
:@*
dtype0

stream_5/conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_namestream_5/conv2d_5/bias
}
*stream_5/conv2d_5/bias/Read/ReadVariableOpReadVariableOpstream_5/conv2d_5/bias*
_output_shapes
:@*
dtype0

stream_6/conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_namestream_6/conv2d_6/kernel

,stream_6/conv2d_6/kernel/Read/ReadVariableOpReadVariableOpstream_6/conv2d_6/kernel*'
_output_shapes
:@*
dtype0

stream_6/conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namestream_6/conv2d_6/bias
~
*stream_6/conv2d_6/bias/Read/ReadVariableOpReadVariableOpstream_6/conv2d_6/bias*
_output_shapes	
:*
dtype0

NoOpNoOp
üV
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*·V
value­VBªV B£V
»
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer_with_weights-7

layer-9
layer-10
layer_with_weights-8
layer-11
layer_with_weights-9
layer-12
layer_with_weights-10
layer-13
	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
R
	variables
trainable_variables
regularization_losses
	keras_api
y
cell
state_shape

states
	variables
trainable_variables
regularization_losses
	keras_api
y
cell
 state_shape

!states
"	variables
#trainable_variables
$regularization_losses
%	keras_api
y
&cell
'state_shape

(states
)	variables
*trainable_variables
+regularization_losses
,	keras_api
y
-cell
.state_shape

/states
0	variables
1trainable_variables
2regularization_losses
3	keras_api
y
4cell
5state_shape

6states
7	variables
8trainable_variables
9regularization_losses
:	keras_api
y
;cell
<state_shape

=states
>	variables
?trainable_variables
@regularization_losses
A	keras_api
y
Bcell
Cstate_shape

Dstates
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
y
Icell
Jstate_shape

Kstates
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
R
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
h

Tkernel
Ubias
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
h

Zkernel
[bias
\	variables
]trainable_variables
^regularization_losses
_	keras_api
h

`kernel
abias
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
Ö
f0
g1
2
h3
i4
!5
j6
k7
(8
l9
m10
/11
n12
o13
614
p15
q16
=17
r18
s19
D20
K21
T22
U23
Z24
[25
`26
a27

f0
g1
h2
i3
j4
k5
l6
m7
n8
o9
p10
q11
r12
s13
T14
U15
Z16
[17
`18
a19
 
­
	variables
tmetrics
ulayer_metrics
vnon_trainable_variables
trainable_variables

wlayers
regularization_losses
xlayer_regularization_losses
 
 
 
 
­
ymetrics
	variables
zlayer_metrics
{non_trainable_variables
trainable_variables

|layers
regularization_losses
}layer_regularization_losses
j

fkernel
gbias
~	variables
trainable_variables
regularization_losses
	keras_api
 
ca
VARIABLE_VALUEstreaming/stream/states6layer_with_weights-0/states/.ATTRIBUTES/VARIABLE_VALUE

f0
g1
2

f0
g1
 
²
metrics
	variables
layer_metrics
non_trainable_variables
trainable_variables
layers
regularization_losses
 layer_regularization_losses
l

hkernel
ibias
	variables
trainable_variables
regularization_losses
	keras_api
 
ec
VARIABLE_VALUEstreaming/stream_1/states6layer_with_weights-1/states/.ATTRIBUTES/VARIABLE_VALUE

h0
i1
!2

h0
i1
 
²
metrics
"	variables
layer_metrics
non_trainable_variables
#trainable_variables
layers
$regularization_losses
 layer_regularization_losses
l

jkernel
kbias
	variables
trainable_variables
regularization_losses
	keras_api
 
ec
VARIABLE_VALUEstreaming/stream_2/states6layer_with_weights-2/states/.ATTRIBUTES/VARIABLE_VALUE

j0
k1
(2

j0
k1
 
²
metrics
)	variables
layer_metrics
non_trainable_variables
*trainable_variables
layers
+regularization_losses
 layer_regularization_losses
l

lkernel
mbias
	variables
trainable_variables
regularization_losses
	keras_api
 
ec
VARIABLE_VALUEstreaming/stream_3/states6layer_with_weights-3/states/.ATTRIBUTES/VARIABLE_VALUE

l0
m1
/2

l0
m1
 
²
metrics
0	variables
layer_metrics
non_trainable_variables
1trainable_variables
 layers
2regularization_losses
 ¡layer_regularization_losses
l

nkernel
obias
¢	variables
£trainable_variables
¤regularization_losses
¥	keras_api
 
ec
VARIABLE_VALUEstreaming/stream_4/states6layer_with_weights-4/states/.ATTRIBUTES/VARIABLE_VALUE

n0
o1
62

n0
o1
 
²
¦metrics
7	variables
§layer_metrics
¨non_trainable_variables
8trainable_variables
©layers
9regularization_losses
 ªlayer_regularization_losses
l

pkernel
qbias
«	variables
¬trainable_variables
­regularization_losses
®	keras_api
 
ec
VARIABLE_VALUEstreaming/stream_5/states6layer_with_weights-5/states/.ATTRIBUTES/VARIABLE_VALUE

p0
q1
=2

p0
q1
 
²
¯metrics
>	variables
°layer_metrics
±non_trainable_variables
?trainable_variables
²layers
@regularization_losses
 ³layer_regularization_losses
l

rkernel
sbias
´	variables
µtrainable_variables
¶regularization_losses
·	keras_api
 
ec
VARIABLE_VALUEstreaming/stream_6/states6layer_with_weights-6/states/.ATTRIBUTES/VARIABLE_VALUE

r0
s1
D2

r0
s1
 
²
¸metrics
E	variables
¹layer_metrics
ºnon_trainable_variables
Ftrainable_variables
»layers
Gregularization_losses
 ¼layer_regularization_losses
V
½	variables
¾trainable_variables
¿regularization_losses
À	keras_api
 
ec
VARIABLE_VALUEstreaming/stream_7/states6layer_with_weights-7/states/.ATTRIBUTES/VARIABLE_VALUE

K0
 
 
²
Ámetrics
L	variables
Âlayer_metrics
Ãnon_trainable_variables
Mtrainable_variables
Älayers
Nregularization_losses
 Ålayer_regularization_losses
 
 
 
²
Æmetrics
P	variables
Çlayer_metrics
Ènon_trainable_variables
Qtrainable_variables
Élayers
Rregularization_losses
 Êlayer_regularization_losses
b`
VARIABLE_VALUEstreaming/dense/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEstreaming/dense/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

T0
U1

T0
U1
 
²
Ëmetrics
V	variables
Ìlayer_metrics
Ínon_trainable_variables
Wtrainable_variables
Îlayers
Xregularization_losses
 Ïlayer_regularization_losses
db
VARIABLE_VALUEstreaming/dense_1/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEstreaming/dense_1/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

Z0
[1

Z0
[1
 
²
Ðmetrics
\	variables
Ñlayer_metrics
Ònon_trainable_variables
]trainable_variables
Ólayers
^regularization_losses
 Ôlayer_regularization_losses
ec
VARIABLE_VALUEstreaming/dense_2/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEstreaming/dense_2/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

`0
a1

`0
a1
 
²
Õmetrics
b	variables
Ölayer_metrics
×non_trainable_variables
ctrainable_variables
Ølayers
dregularization_losses
 Ùlayer_regularization_losses
PN
VARIABLE_VALUEstream/conv2d/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEstream/conv2d/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEstream_1/conv2d_1/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEstream_1/conv2d_1/bias&variables/4/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEstream_2/conv2d_2/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEstream_2/conv2d_2/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEstream_3/conv2d_3/kernel&variables/9/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEstream_3/conv2d_3/bias'variables/10/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEstream_4/conv2d_4/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEstream_4/conv2d_4/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEstream_5/conv2d_5/kernel'variables/15/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEstream_5/conv2d_5/bias'variables/16/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEstream_6/conv2d_6/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEstream_6/conv2d_6/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE
 
 
8
0
!1
(2
/3
64
=5
D6
K7
f
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
 
 
 
 
 
 

f0
g1

f0
g1
 
³
Úmetrics
~	variables
Ûlayer_metrics
Ünon_trainable_variables
trainable_variables
Ýlayers
regularization_losses
 Þlayer_regularization_losses
 
 

0

0
 

h0
i1

h0
i1
 
µ
ßmetrics
	variables
àlayer_metrics
ánon_trainable_variables
trainable_variables
âlayers
regularization_losses
 ãlayer_regularization_losses
 
 

!0

0
 

j0
k1

j0
k1
 
µ
ämetrics
	variables
ålayer_metrics
ænon_trainable_variables
trainable_variables
çlayers
regularization_losses
 èlayer_regularization_losses
 
 

(0

&0
 

l0
m1

l0
m1
 
µ
émetrics
	variables
êlayer_metrics
ënon_trainable_variables
trainable_variables
ìlayers
regularization_losses
 ílayer_regularization_losses
 
 

/0

-0
 

n0
o1

n0
o1
 
µ
îmetrics
¢	variables
ïlayer_metrics
ðnon_trainable_variables
£trainable_variables
ñlayers
¤regularization_losses
 òlayer_regularization_losses
 
 

60

40
 

p0
q1

p0
q1
 
µ
ómetrics
«	variables
ôlayer_metrics
õnon_trainable_variables
¬trainable_variables
ölayers
­regularization_losses
 ÷layer_regularization_losses
 
 

=0

;0
 

r0
s1

r0
s1
 
µ
ømetrics
´	variables
ùlayer_metrics
únon_trainable_variables
µtrainable_variables
ûlayers
¶regularization_losses
 ülayer_regularization_losses
 
 

D0

B0
 
 
 
 
µ
ýmetrics
½	variables
þlayer_metrics
ÿnon_trainable_variables
¾trainable_variables
layers
¿regularization_losses
 layer_regularization_losses
 
 

K0

I0
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
t
serving_default_input_audioPlaceholder*"
_output_shapes
:*
dtype0*
shape:

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_audiostreaming/stream/statesstream/conv2d/kernelstream/conv2d/biasstreaming/stream_1/statesstream_1/conv2d_1/kernelstream_1/conv2d_1/biasstreaming/stream_2/statesstream_2/conv2d_2/kernelstream_2/conv2d_2/biasstreaming/stream_3/statesstream_3/conv2d_3/kernelstream_3/conv2d_3/biasstreaming/stream_4/statesstream_4/conv2d_4/kernelstream_4/conv2d_4/biasstreaming/stream_5/statesstream_5/conv2d_5/kernelstream_5/conv2d_5/biasstreaming/stream_6/statesstream_6/conv2d_6/kernelstream_6/conv2d_6/biasstreaming/stream_7/statesstreaming/dense/kernelstreaming/dense/biasstreaming/dense_1/kernelstreaming/dense_1/biasstreaming/dense_2/kernelstreaming/dense_2/bias*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*6
_read_only_resource_inputs
	**
config_proto

CPU

GPU 2J 8*+
f&R$
"__inference_signature_wrapper_3423
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
 
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename+streaming/stream/states/Read/ReadVariableOp-streaming/stream_1/states/Read/ReadVariableOp-streaming/stream_2/states/Read/ReadVariableOp-streaming/stream_3/states/Read/ReadVariableOp-streaming/stream_4/states/Read/ReadVariableOp-streaming/stream_5/states/Read/ReadVariableOp-streaming/stream_6/states/Read/ReadVariableOp-streaming/stream_7/states/Read/ReadVariableOp*streaming/dense/kernel/Read/ReadVariableOp(streaming/dense/bias/Read/ReadVariableOp,streaming/dense_1/kernel/Read/ReadVariableOp*streaming/dense_1/bias/Read/ReadVariableOp,streaming/dense_2/kernel/Read/ReadVariableOp*streaming/dense_2/bias/Read/ReadVariableOp(stream/conv2d/kernel/Read/ReadVariableOp&stream/conv2d/bias/Read/ReadVariableOp,stream_1/conv2d_1/kernel/Read/ReadVariableOp*stream_1/conv2d_1/bias/Read/ReadVariableOp,stream_2/conv2d_2/kernel/Read/ReadVariableOp*stream_2/conv2d_2/bias/Read/ReadVariableOp,stream_3/conv2d_3/kernel/Read/ReadVariableOp*stream_3/conv2d_3/bias/Read/ReadVariableOp,stream_4/conv2d_4/kernel/Read/ReadVariableOp*stream_4/conv2d_4/bias/Read/ReadVariableOp,stream_5/conv2d_5/kernel/Read/ReadVariableOp*stream_5/conv2d_5/bias/Read/ReadVariableOp,stream_6/conv2d_6/kernel/Read/ReadVariableOp*stream_6/conv2d_6/bias/Read/ReadVariableOpConst*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*&
f!R
__inference__traced_save_4543
ë
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamestreaming/stream/statesstreaming/stream_1/statesstreaming/stream_2/statesstreaming/stream_3/statesstreaming/stream_4/statesstreaming/stream_5/statesstreaming/stream_6/statesstreaming/stream_7/statesstreaming/dense/kernelstreaming/dense/biasstreaming/dense_1/kernelstreaming/dense_1/biasstreaming/dense_2/kernelstreaming/dense_2/biasstream/conv2d/kernelstream/conv2d/biasstream_1/conv2d_1/kernelstream_1/conv2d_1/biasstream_2/conv2d_2/kernelstream_2/conv2d_2/biasstream_3/conv2d_3/kernelstream_3/conv2d_3/biasstream_4/conv2d_4/kernelstream_4/conv2d_4/biasstream_5/conv2d_5/kernelstream_5/conv2d_5/biasstream_6/conv2d_6/kernelstream_6/conv2d_6/bias*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*)
f$R"
 __inference__traced_restore_4639
Á

@__inference_stream_layer_call_and_return_conditional_losses_4163

inputs*
&readvariableop_streaming_stream_states5
1conv2d_conv2d_readvariableop_stream_conv2d_kernel4
0conv2d_biasadd_readvariableop_stream_conv2d_bias
identity¢AssignVariableOp
ReadVariableOpReadVariableOp&readvariableop_streaming_stream_states*&
_output_shapes
:*
dtype02
ReadVariableOp
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:*

begin_mask*
end_mask2
strided_slice\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2strided_slice:output:0inputsconcat/axis:output:0*
N*
T0*&
_output_shapes
:2
concat¥
AssignVariableOpAssignVariableOp&readvariableop_streaming_stream_statesconcat:output:0^ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOpÉ
conv2d/Conv2D/ReadVariableOpReadVariableOp1conv2d_conv2d_readvariableop_stream_conv2d_kernel^AssignVariableOp*&
_output_shapes
:@*
dtype02
conv2d/Conv2D/ReadVariableOp¹
conv2d/Conv2DConv2Dconcat:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
conv2d/Conv2D¾
conv2d/BiasAdd/ReadVariableOpReadVariableOp0conv2d_biasadd_readvariableop_stream_conv2d_bias^AssignVariableOp*
_output_shapes
:@*
dtype02
conv2d/BiasAdd/ReadVariableOp
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
conv2d/BiasAddl
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*&
_output_shapes
:@2
conv2d/Relu
IdentityIdentityconv2d/Relu:activations:0^AssignVariableOp*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*1
_input_shapes 
::::2$
AssignVariableOpAssignVariableOp:N J
&
_output_shapes
:
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ä
È
B__inference_conv2d_4_layer_call_and_return_conditional_losses_2722

inputs2
.conv2d_readvariableop_stream_4_conv2d_4_kernel1
-biasadd_readvariableop_stream_4_conv2d_4_bias
identityo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate¦
Conv2D/ReadVariableOpReadVariableOp.conv2d_readvariableop_stream_4_conv2d_4_kernel*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp·
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOp-biasadd_readvariableop_stream_4_conv2d_4_bias*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
¸

'__inference_conv2d_6_layer_call_fn_2808

inputs
stream_6_conv2d_6_kernel
stream_6_conv2d_6_bias
identity¢StatefulPartitionedCall¨
StatefulPartitionedCallStatefulPartitionedCallinputsstream_6_conv2d_6_kernelstream_6_conv2d_6_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_conv2d_6_layer_call_and_return_conditional_losses_28032
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
á
È
B__inference_conv2d_5_layer_call_and_return_conditional_losses_2755

inputs2
.conv2d_readvariableop_stream_5_conv2d_5_kernel1
-biasadd_readvariableop_stream_5_conv2d_5_bias
identityo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate¦
Conv2D/ReadVariableOpReadVariableOp.conv2d_readvariableop_stream_5_conv2d_5_kernel*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp¶
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOp-biasadd_readvariableop_stream_5_conv2d_5_bias*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2
Relu
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ÏM
ù
F__inference_functional_1_layer_call_and_return_conditional_losses_3357

inputs"
stream_streaming_stream_states
stream_stream_conv2d_kernel
stream_stream_conv2d_bias&
"stream_1_streaming_stream_1_states%
!stream_1_stream_1_conv2d_1_kernel#
stream_1_stream_1_conv2d_1_bias&
"stream_2_streaming_stream_2_states%
!stream_2_stream_2_conv2d_2_kernel#
stream_2_stream_2_conv2d_2_bias&
"stream_3_streaming_stream_3_states%
!stream_3_stream_3_conv2d_3_kernel#
stream_3_stream_3_conv2d_3_bias&
"stream_4_streaming_stream_4_states%
!stream_4_stream_4_conv2d_4_kernel#
stream_4_stream_4_conv2d_4_bias&
"stream_5_streaming_stream_5_states%
!stream_5_stream_5_conv2d_5_kernel#
stream_5_stream_5_conv2d_5_bias&
"stream_6_streaming_stream_6_states%
!stream_6_stream_6_conv2d_6_kernel#
stream_6_stream_6_conv2d_6_bias&
"stream_7_streaming_stream_7_states 
dense_streaming_dense_kernel
dense_streaming_dense_bias$
 dense_1_streaming_dense_1_kernel"
dense_1_streaming_dense_1_bias$
 dense_2_streaming_dense_2_kernel"
dense_2_streaming_dense_2_bias
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢stream/StatefulPartitionedCall¢ stream_1/StatefulPartitionedCall¢ stream_2/StatefulPartitionedCall¢ stream_3/StatefulPartitionedCall¢ stream_4/StatefulPartitionedCall¢ stream_5/StatefulPartitionedCall¢ stream_6/StatefulPartitionedCall¢ stream_7/StatefulPartitionedCallø
&tf_op_layer_ExpandDims/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*Y
fTRR
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_28182(
&tf_op_layer_ExpandDims/PartitionedCallè
stream/StatefulPartitionedCallStatefulPartitionedCall/tf_op_layer_ExpandDims/PartitionedCall:output:0stream_streaming_stream_statesstream_stream_conv2d_kernelstream_stream_conv2d_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_stream_layer_call_and_return_conditional_losses_28462 
stream/StatefulPartitionedCallö
 stream_1/StatefulPartitionedCallStatefulPartitionedCall'stream/StatefulPartitionedCall:output:0"stream_1_streaming_stream_1_states!stream_1_stream_1_conv2d_1_kernelstream_1_stream_1_conv2d_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_stream_1_layer_call_and_return_conditional_losses_28802"
 stream_1/StatefulPartitionedCallø
 stream_2/StatefulPartitionedCallStatefulPartitionedCall)stream_1/StatefulPartitionedCall:output:0"stream_2_streaming_stream_2_states!stream_2_stream_2_conv2d_2_kernelstream_2_stream_2_conv2d_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_stream_2_layer_call_and_return_conditional_losses_29142"
 stream_2/StatefulPartitionedCallø
 stream_3/StatefulPartitionedCallStatefulPartitionedCall)stream_2/StatefulPartitionedCall:output:0"stream_3_streaming_stream_3_states!stream_3_stream_3_conv2d_3_kernelstream_3_stream_3_conv2d_3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_stream_3_layer_call_and_return_conditional_losses_29482"
 stream_3/StatefulPartitionedCallù
 stream_4/StatefulPartitionedCallStatefulPartitionedCall)stream_3/StatefulPartitionedCall:output:0"stream_4_streaming_stream_4_states!stream_4_stream_4_conv2d_4_kernelstream_4_stream_4_conv2d_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_stream_4_layer_call_and_return_conditional_losses_29822"
 stream_4/StatefulPartitionedCallø
 stream_5/StatefulPartitionedCallStatefulPartitionedCall)stream_4/StatefulPartitionedCall:output:0"stream_5_streaming_stream_5_states!stream_5_stream_5_conv2d_5_kernelstream_5_stream_5_conv2d_5_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_stream_5_layer_call_and_return_conditional_losses_30162"
 stream_5/StatefulPartitionedCallù
 stream_6/StatefulPartitionedCallStatefulPartitionedCall)stream_5/StatefulPartitionedCall:output:0"stream_6_streaming_stream_6_states!stream_6_stream_6_conv2d_6_kernelstream_6_stream_6_conv2d_6_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_stream_6_layer_call_and_return_conditional_losses_30502"
 stream_6/StatefulPartitionedCall§
 stream_7/StatefulPartitionedCallStatefulPartitionedCall)stream_6/StatefulPartitionedCall:output:0"stream_7_streaming_stream_7_states*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_stream_7_layer_call_and_return_conditional_losses_30782"
 stream_7/StatefulPartitionedCallç
dropout/PartitionedCallPartitionedCall)stream_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_31052
dropout/PartitionedCall°
dense/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_streaming_dense_kerneldense_streaming_dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_31282
dense/StatefulPartitionedCallÄ
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0 dense_1_streaming_dense_1_kerneldense_1_streaming_dense_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_31512!
dense_1/StatefulPartitionedCallÅ
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0 dense_2_streaming_dense_2_kerneldense_2_streaming_dense_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_31732!
dense_2/StatefulPartitionedCallí
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^stream/StatefulPartitionedCall!^stream_1/StatefulPartitionedCall!^stream_2/StatefulPartitionedCall!^stream_3/StatefulPartitionedCall!^stream_4/StatefulPartitionedCall!^stream_5/StatefulPartitionedCall!^stream_6/StatefulPartitionedCall!^stream_7/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*
_input_shapes
~:::::::::::::::::::::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2@
stream/StatefulPartitionedCallstream/StatefulPartitionedCall2D
 stream_1/StatefulPartitionedCall stream_1/StatefulPartitionedCall2D
 stream_2/StatefulPartitionedCall stream_2/StatefulPartitionedCall2D
 stream_3/StatefulPartitionedCall stream_3/StatefulPartitionedCall2D
 stream_4/StatefulPartitionedCall stream_4/StatefulPartitionedCall2D
 stream_5/StatefulPartitionedCall stream_5/StatefulPartitionedCall2D
 stream_6/StatefulPartitionedCall stream_6/StatefulPartitionedCall2D
 stream_7/StatefulPartitionedCall stream_7/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
¸

'__inference_conv2d_5_layer_call_fn_2775

inputs
stream_5_conv2d_5_kernel
stream_5_conv2d_5_bias
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsstream_5_conv2d_5_kernelstream_5_conv2d_5_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_conv2d_5_layer_call_and_return_conditional_losses_27702
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Þ
È
B__inference_conv2d_3_layer_call_and_return_conditional_losses_2704

inputs2
.conv2d_readvariableop_stream_3_conv2d_3_kernel1
-biasadd_readvariableop_stream_3_conv2d_3_bias
identityo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate¥
Conv2D/ReadVariableOpReadVariableOp.conv2d_readvariableop_stream_3_conv2d_3_kernel*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¶
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOp-biasadd_readvariableop_stream_3_conv2d_3_bias*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2
Relu
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 


B__inference_stream_4_layer_call_and_return_conditional_losses_2982

inputs,
(readvariableop_streaming_stream_4_states;
7conv2d_4_conv2d_readvariableop_stream_4_conv2d_4_kernel:
6conv2d_4_biasadd_readvariableop_stream_4_conv2d_4_bias
identity¢AssignVariableOp
ReadVariableOpReadVariableOp(readvariableop_streaming_stream_4_states*&
_output_shapes
:@*
dtype02
ReadVariableOp
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:@*

begin_mask*
end_mask2
strided_slice\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2strided_slice:output:0inputsconcat/axis:output:0*
N*
T0*&
_output_shapes
:@2
concat§
AssignVariableOpAssignVariableOp(readvariableop_streaming_stream_4_statesconcat:output:0^ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOp
conv2d_4/dilation_rateConst^AssignVariableOp*
_output_shapes
:*
dtype0*
valueB"      2
conv2d_4/dilation_rateÔ
conv2d_4/Conv2D/ReadVariableOpReadVariableOp7conv2d_4_conv2d_readvariableop_stream_4_conv2d_4_kernel^AssignVariableOp*'
_output_shapes
:@*
dtype02 
conv2d_4/Conv2D/ReadVariableOpÀ
conv2d_4/Conv2DConv2Dconcat:output:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:*
paddingVALID*
strides
2
conv2d_4/Conv2DÉ
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp6conv2d_4_biasadd_readvariableop_stream_4_conv2d_4_bias^AssignVariableOp*
_output_shapes	
:*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp¤
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:2
conv2d_4/BiasAdds
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*'
_output_shapes
:2
conv2d_4/Relu
IdentityIdentityconv2d_4/Relu:activations:0^AssignVariableOp*
T0*'
_output_shapes
:2

Identity"
identityIdentity:output:0*1
_input_shapes 
:@:::2$
AssignVariableOpAssignVariableOp:N J
&
_output_shapes
:@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
¢

%__inference_conv2d_layer_call_fn_2610

inputs
stream_conv2d_kernel
stream_conv2d_bias
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstream_conv2d_kernelstream_conv2d_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_26052
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Þ
È
B__inference_conv2d_2_layer_call_and_return_conditional_losses_2656

inputs2
.conv2d_readvariableop_stream_2_conv2d_2_kernel1
-biasadd_readvariableop_stream_2_conv2d_2_bias
identityo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate¥
Conv2D/ReadVariableOpReadVariableOp.conv2d_readvariableop_stream_2_conv2d_2_kernel*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¶
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOp-biasadd_readvariableop_stream_2_conv2d_2_bias*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2
Relu
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
¤
_
A__inference_dropout_layer_call_and_return_conditional_losses_4370

inputs

identity_1R
IdentityIdentityinputs*
T0*
_output_shapes
:	2

Identitya

Identity_1IdentityIdentity:output:0*
T0*
_output_shapes
:	2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs
§
¹
'__inference_stream_2_layer_call_fn_4225

inputs
streaming_stream_2_states
stream_2_conv2d_2_kernel
stream_2_conv2d_2_bias
identity¢StatefulPartitionedCall¨
StatefulPartitionedCallStatefulPartitionedCallinputsstreaming_stream_2_statesstream_2_conv2d_2_kernelstream_2_conv2d_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_stream_2_layer_call_and_return_conditional_losses_29142
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*1
_input_shapes 
:@:::22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 


B__inference_stream_5_layer_call_and_return_conditional_losses_3016

inputs,
(readvariableop_streaming_stream_5_states;
7conv2d_5_conv2d_readvariableop_stream_5_conv2d_5_kernel:
6conv2d_5_biasadd_readvariableop_stream_5_conv2d_5_bias
identity¢AssignVariableOp
ReadVariableOpReadVariableOp(readvariableop_streaming_stream_5_states*'
_output_shapes
:*
dtype02
ReadVariableOp
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:*

begin_mask*
end_mask2
strided_slice\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2strided_slice:output:0inputsconcat/axis:output:0*
N*
T0*'
_output_shapes
:2
concat§
AssignVariableOpAssignVariableOp(readvariableop_streaming_stream_5_statesconcat:output:0^ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOp
conv2d_5/dilation_rateConst^AssignVariableOp*
_output_shapes
:*
dtype0*
valueB"      2
conv2d_5/dilation_rateÔ
conv2d_5/Conv2D/ReadVariableOpReadVariableOp7conv2d_5_conv2d_readvariableop_stream_5_conv2d_5_kernel^AssignVariableOp*'
_output_shapes
:@*
dtype02 
conv2d_5/Conv2D/ReadVariableOp¿
conv2d_5/Conv2DConv2Dconcat:output:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
conv2d_5/Conv2DÈ
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp6conv2d_5_biasadd_readvariableop_stream_5_conv2d_5_bias^AssignVariableOp*
_output_shapes
:@*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp£
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
conv2d_5/BiasAddr
conv2d_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*&
_output_shapes
:@2
conv2d_5/Relu
IdentityIdentityconv2d_5/Relu:activations:0^AssignVariableOp*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*2
_input_shapes!
::::2$
AssignVariableOpAssignVariableOp:O K
'
_output_shapes
:
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ª

&__inference_dense_2_layer_call_fn_4432

inputs
streaming_dense_2_kernel
streaming_dense_2_bias
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstreaming_dense_2_kernelstreaming_dense_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_31732
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
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Ô

F__inference_functional_1_layer_call_and_return_conditional_losses_3571

inputs1
-stream_readvariableop_streaming_stream_states<
8stream_conv2d_conv2d_readvariableop_stream_conv2d_kernel;
7stream_conv2d_biasadd_readvariableop_stream_conv2d_bias5
1stream_1_readvariableop_streaming_stream_1_statesD
@stream_1_conv2d_1_conv2d_readvariableop_stream_1_conv2d_1_kernelC
?stream_1_conv2d_1_biasadd_readvariableop_stream_1_conv2d_1_bias5
1stream_2_readvariableop_streaming_stream_2_statesD
@stream_2_conv2d_2_conv2d_readvariableop_stream_2_conv2d_2_kernelC
?stream_2_conv2d_2_biasadd_readvariableop_stream_2_conv2d_2_bias5
1stream_3_readvariableop_streaming_stream_3_statesD
@stream_3_conv2d_3_conv2d_readvariableop_stream_3_conv2d_3_kernelC
?stream_3_conv2d_3_biasadd_readvariableop_stream_3_conv2d_3_bias5
1stream_4_readvariableop_streaming_stream_4_statesD
@stream_4_conv2d_4_conv2d_readvariableop_stream_4_conv2d_4_kernelC
?stream_4_conv2d_4_biasadd_readvariableop_stream_4_conv2d_4_bias5
1stream_5_readvariableop_streaming_stream_5_statesD
@stream_5_conv2d_5_conv2d_readvariableop_stream_5_conv2d_5_kernelC
?stream_5_conv2d_5_biasadd_readvariableop_stream_5_conv2d_5_bias5
1stream_6_readvariableop_streaming_stream_6_statesD
@stream_6_conv2d_6_conv2d_readvariableop_stream_6_conv2d_6_kernelC
?stream_6_conv2d_6_biasadd_readvariableop_stream_6_conv2d_6_bias5
1stream_7_readvariableop_streaming_stream_7_states6
2dense_matmul_readvariableop_streaming_dense_kernel5
1dense_biasadd_readvariableop_streaming_dense_bias:
6dense_1_matmul_readvariableop_streaming_dense_1_kernel9
5dense_1_biasadd_readvariableop_streaming_dense_1_bias:
6dense_2_matmul_readvariableop_streaming_dense_2_kernel9
5dense_2_biasadd_readvariableop_streaming_dense_2_bias
identity¢stream/AssignVariableOp¢stream_1/AssignVariableOp¢stream_2/AssignVariableOp¢stream_3/AssignVariableOp¢stream_4/AssignVariableOp¢stream_5/AssignVariableOp¢stream_6/AssignVariableOp¢stream_7/AssignVariableOp
%tf_op_layer_ExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2'
%tf_op_layer_ExpandDims/ExpandDims/dimÌ
!tf_op_layer_ExpandDims/ExpandDims
ExpandDimsinputs.tf_op_layer_ExpandDims/ExpandDims/dim:output:0*
T0*
_cloned(*&
_output_shapes
:2#
!tf_op_layer_ExpandDims/ExpandDims¤
stream/ReadVariableOpReadVariableOp-stream_readvariableop_streaming_stream_states*&
_output_shapes
:*
dtype02
stream/ReadVariableOp
stream/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
stream/strided_slice/stack
stream/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
stream/strided_slice/stack_1
stream/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
stream/strided_slice/stack_2®
stream/strided_sliceStridedSlicestream/ReadVariableOp:value:0#stream/strided_slice/stack:output:0%stream/strided_slice/stack_1:output:0%stream/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:*

begin_mask*
end_mask2
stream/strided_slicej
stream/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
stream/concat/axisÌ
stream/concatConcatV2stream/strided_slice:output:0*tf_op_layer_ExpandDims/ExpandDims:output:0stream/concat/axis:output:0*
N*
T0*&
_output_shapes
:2
stream/concatÈ
stream/AssignVariableOpAssignVariableOp-stream_readvariableop_streaming_stream_statesstream/concat:output:0^stream/ReadVariableOp*
_output_shapes
 *
dtype02
stream/AssignVariableOpå
#stream/conv2d/Conv2D/ReadVariableOpReadVariableOp8stream_conv2d_conv2d_readvariableop_stream_conv2d_kernel^stream/AssignVariableOp*&
_output_shapes
:@*
dtype02%
#stream/conv2d/Conv2D/ReadVariableOpÕ
stream/conv2d/Conv2DConv2Dstream/concat:output:0+stream/conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
stream/conv2d/Conv2DÚ
$stream/conv2d/BiasAdd/ReadVariableOpReadVariableOp7stream_conv2d_biasadd_readvariableop_stream_conv2d_bias^stream/AssignVariableOp*
_output_shapes
:@*
dtype02&
$stream/conv2d/BiasAdd/ReadVariableOp·
stream/conv2d/BiasAddBiasAddstream/conv2d/Conv2D:output:0,stream/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
stream/conv2d/BiasAdd
stream/conv2d/ReluRelustream/conv2d/BiasAdd:output:0*
T0*&
_output_shapes
:@2
stream/conv2d/Relu¬
stream_1/ReadVariableOpReadVariableOp1stream_1_readvariableop_streaming_stream_1_states*&
_output_shapes
:@*
dtype02
stream_1/ReadVariableOp
stream_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
stream_1/strided_slice/stack
stream_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2 
stream_1/strided_slice/stack_1
stream_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2 
stream_1/strided_slice/stack_2º
stream_1/strided_sliceStridedSlicestream_1/ReadVariableOp:value:0%stream_1/strided_slice/stack:output:0'stream_1/strided_slice/stack_1:output:0'stream_1/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:@*

begin_mask*
end_mask2
stream_1/strided_slicen
stream_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
stream_1/concat/axisÊ
stream_1/concatConcatV2stream_1/strided_slice:output:0 stream/conv2d/Relu:activations:0stream_1/concat/axis:output:0*
N*
T0*&
_output_shapes
:@2
stream_1/concatÔ
stream_1/AssignVariableOpAssignVariableOp1stream_1_readvariableop_streaming_stream_1_statesstream_1/concat:output:0^stream_1/ReadVariableOp*
_output_shapes
 *
dtype02
stream_1/AssignVariableOp÷
'stream_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp@stream_1_conv2d_1_conv2d_readvariableop_stream_1_conv2d_1_kernel^stream_1/AssignVariableOp*&
_output_shapes
:@@*
dtype02)
'stream_1/conv2d_1/Conv2D/ReadVariableOpã
stream_1/conv2d_1/Conv2DConv2Dstream_1/concat:output:0/stream_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
stream_1/conv2d_1/Conv2Dì
(stream_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp?stream_1_conv2d_1_biasadd_readvariableop_stream_1_conv2d_1_bias^stream_1/AssignVariableOp*
_output_shapes
:@*
dtype02*
(stream_1/conv2d_1/BiasAdd/ReadVariableOpÇ
stream_1/conv2d_1/BiasAddBiasAdd!stream_1/conv2d_1/Conv2D:output:00stream_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
stream_1/conv2d_1/BiasAdd
stream_1/conv2d_1/ReluRelu"stream_1/conv2d_1/BiasAdd:output:0*
T0*&
_output_shapes
:@2
stream_1/conv2d_1/Relu¬
stream_2/ReadVariableOpReadVariableOp1stream_2_readvariableop_streaming_stream_2_states*&
_output_shapes
:@*
dtype02
stream_2/ReadVariableOp
stream_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
stream_2/strided_slice/stack
stream_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2 
stream_2/strided_slice/stack_1
stream_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2 
stream_2/strided_slice/stack_2º
stream_2/strided_sliceStridedSlicestream_2/ReadVariableOp:value:0%stream_2/strided_slice/stack:output:0'stream_2/strided_slice/stack_1:output:0'stream_2/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:@*

begin_mask*
end_mask2
stream_2/strided_slicen
stream_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
stream_2/concat/axisÎ
stream_2/concatConcatV2stream_2/strided_slice:output:0$stream_1/conv2d_1/Relu:activations:0stream_2/concat/axis:output:0*
N*
T0*&
_output_shapes
:@2
stream_2/concatÔ
stream_2/AssignVariableOpAssignVariableOp1stream_2_readvariableop_streaming_stream_2_statesstream_2/concat:output:0^stream_2/ReadVariableOp*
_output_shapes
 *
dtype02
stream_2/AssignVariableOp÷
'stream_2/conv2d_2/Conv2D/ReadVariableOpReadVariableOp@stream_2_conv2d_2_conv2d_readvariableop_stream_2_conv2d_2_kernel^stream_2/AssignVariableOp*&
_output_shapes
:@@*
dtype02)
'stream_2/conv2d_2/Conv2D/ReadVariableOpã
stream_2/conv2d_2/Conv2DConv2Dstream_2/concat:output:0/stream_2/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
stream_2/conv2d_2/Conv2Dì
(stream_2/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp?stream_2_conv2d_2_biasadd_readvariableop_stream_2_conv2d_2_bias^stream_2/AssignVariableOp*
_output_shapes
:@*
dtype02*
(stream_2/conv2d_2/BiasAdd/ReadVariableOpÇ
stream_2/conv2d_2/BiasAddBiasAdd!stream_2/conv2d_2/Conv2D:output:00stream_2/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
stream_2/conv2d_2/BiasAdd
stream_2/conv2d_2/ReluRelu"stream_2/conv2d_2/BiasAdd:output:0*
T0*&
_output_shapes
:@2
stream_2/conv2d_2/Relu¬
stream_3/ReadVariableOpReadVariableOp1stream_3_readvariableop_streaming_stream_3_states*&
_output_shapes
:@*
dtype02
stream_3/ReadVariableOp
stream_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
stream_3/strided_slice/stack
stream_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2 
stream_3/strided_slice/stack_1
stream_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2 
stream_3/strided_slice/stack_2º
stream_3/strided_sliceStridedSlicestream_3/ReadVariableOp:value:0%stream_3/strided_slice/stack:output:0'stream_3/strided_slice/stack_1:output:0'stream_3/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:@*

begin_mask*
end_mask2
stream_3/strided_slicen
stream_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
stream_3/concat/axisÎ
stream_3/concatConcatV2stream_3/strided_slice:output:0$stream_2/conv2d_2/Relu:activations:0stream_3/concat/axis:output:0*
N*
T0*&
_output_shapes
:@2
stream_3/concatÔ
stream_3/AssignVariableOpAssignVariableOp1stream_3_readvariableop_streaming_stream_3_statesstream_3/concat:output:0^stream_3/ReadVariableOp*
_output_shapes
 *
dtype02
stream_3/AssignVariableOp÷
'stream_3/conv2d_3/Conv2D/ReadVariableOpReadVariableOp@stream_3_conv2d_3_conv2d_readvariableop_stream_3_conv2d_3_kernel^stream_3/AssignVariableOp*&
_output_shapes
:@@*
dtype02)
'stream_3/conv2d_3/Conv2D/ReadVariableOpã
stream_3/conv2d_3/Conv2DConv2Dstream_3/concat:output:0/stream_3/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
stream_3/conv2d_3/Conv2Dì
(stream_3/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp?stream_3_conv2d_3_biasadd_readvariableop_stream_3_conv2d_3_bias^stream_3/AssignVariableOp*
_output_shapes
:@*
dtype02*
(stream_3/conv2d_3/BiasAdd/ReadVariableOpÇ
stream_3/conv2d_3/BiasAddBiasAdd!stream_3/conv2d_3/Conv2D:output:00stream_3/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
stream_3/conv2d_3/BiasAdd
stream_3/conv2d_3/ReluRelu"stream_3/conv2d_3/BiasAdd:output:0*
T0*&
_output_shapes
:@2
stream_3/conv2d_3/Relu¬
stream_4/ReadVariableOpReadVariableOp1stream_4_readvariableop_streaming_stream_4_states*&
_output_shapes
:@*
dtype02
stream_4/ReadVariableOp
stream_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
stream_4/strided_slice/stack
stream_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2 
stream_4/strided_slice/stack_1
stream_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2 
stream_4/strided_slice/stack_2º
stream_4/strided_sliceStridedSlicestream_4/ReadVariableOp:value:0%stream_4/strided_slice/stack:output:0'stream_4/strided_slice/stack_1:output:0'stream_4/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:@*

begin_mask*
end_mask2
stream_4/strided_slicen
stream_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
stream_4/concat/axisÎ
stream_4/concatConcatV2stream_4/strided_slice:output:0$stream_3/conv2d_3/Relu:activations:0stream_4/concat/axis:output:0*
N*
T0*&
_output_shapes
:@2
stream_4/concatÔ
stream_4/AssignVariableOpAssignVariableOp1stream_4_readvariableop_streaming_stream_4_statesstream_4/concat:output:0^stream_4/ReadVariableOp*
_output_shapes
 *
dtype02
stream_4/AssignVariableOpø
'stream_4/conv2d_4/Conv2D/ReadVariableOpReadVariableOp@stream_4_conv2d_4_conv2d_readvariableop_stream_4_conv2d_4_kernel^stream_4/AssignVariableOp*'
_output_shapes
:@*
dtype02)
'stream_4/conv2d_4/Conv2D/ReadVariableOpä
stream_4/conv2d_4/Conv2DConv2Dstream_4/concat:output:0/stream_4/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:*
paddingVALID*
strides
2
stream_4/conv2d_4/Conv2Dí
(stream_4/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp?stream_4_conv2d_4_biasadd_readvariableop_stream_4_conv2d_4_bias^stream_4/AssignVariableOp*
_output_shapes	
:*
dtype02*
(stream_4/conv2d_4/BiasAdd/ReadVariableOpÈ
stream_4/conv2d_4/BiasAddBiasAdd!stream_4/conv2d_4/Conv2D:output:00stream_4/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:2
stream_4/conv2d_4/BiasAdd
stream_4/conv2d_4/ReluRelu"stream_4/conv2d_4/BiasAdd:output:0*
T0*'
_output_shapes
:2
stream_4/conv2d_4/Relu­
stream_5/ReadVariableOpReadVariableOp1stream_5_readvariableop_streaming_stream_5_states*'
_output_shapes
:*
dtype02
stream_5/ReadVariableOp
stream_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
stream_5/strided_slice/stack
stream_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2 
stream_5/strided_slice/stack_1
stream_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2 
stream_5/strided_slice/stack_2»
stream_5/strided_sliceStridedSlicestream_5/ReadVariableOp:value:0%stream_5/strided_slice/stack:output:0'stream_5/strided_slice/stack_1:output:0'stream_5/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:*

begin_mask*
end_mask2
stream_5/strided_slicen
stream_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
stream_5/concat/axisÏ
stream_5/concatConcatV2stream_5/strided_slice:output:0$stream_4/conv2d_4/Relu:activations:0stream_5/concat/axis:output:0*
N*
T0*'
_output_shapes
:2
stream_5/concatÔ
stream_5/AssignVariableOpAssignVariableOp1stream_5_readvariableop_streaming_stream_5_statesstream_5/concat:output:0^stream_5/ReadVariableOp*
_output_shapes
 *
dtype02
stream_5/AssignVariableOpø
'stream_5/conv2d_5/Conv2D/ReadVariableOpReadVariableOp@stream_5_conv2d_5_conv2d_readvariableop_stream_5_conv2d_5_kernel^stream_5/AssignVariableOp*'
_output_shapes
:@*
dtype02)
'stream_5/conv2d_5/Conv2D/ReadVariableOpã
stream_5/conv2d_5/Conv2DConv2Dstream_5/concat:output:0/stream_5/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
stream_5/conv2d_5/Conv2Dì
(stream_5/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp?stream_5_conv2d_5_biasadd_readvariableop_stream_5_conv2d_5_bias^stream_5/AssignVariableOp*
_output_shapes
:@*
dtype02*
(stream_5/conv2d_5/BiasAdd/ReadVariableOpÇ
stream_5/conv2d_5/BiasAddBiasAdd!stream_5/conv2d_5/Conv2D:output:00stream_5/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
stream_5/conv2d_5/BiasAdd
stream_5/conv2d_5/ReluRelu"stream_5/conv2d_5/BiasAdd:output:0*
T0*&
_output_shapes
:@2
stream_5/conv2d_5/Relu¬
stream_6/ReadVariableOpReadVariableOp1stream_6_readvariableop_streaming_stream_6_states*&
_output_shapes
:@*
dtype02
stream_6/ReadVariableOp
stream_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
stream_6/strided_slice/stack
stream_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2 
stream_6/strided_slice/stack_1
stream_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2 
stream_6/strided_slice/stack_2º
stream_6/strided_sliceStridedSlicestream_6/ReadVariableOp:value:0%stream_6/strided_slice/stack:output:0'stream_6/strided_slice/stack_1:output:0'stream_6/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:@*

begin_mask*
end_mask2
stream_6/strided_slicen
stream_6/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
stream_6/concat/axisÎ
stream_6/concatConcatV2stream_6/strided_slice:output:0$stream_5/conv2d_5/Relu:activations:0stream_6/concat/axis:output:0*
N*
T0*&
_output_shapes
:@2
stream_6/concatÔ
stream_6/AssignVariableOpAssignVariableOp1stream_6_readvariableop_streaming_stream_6_statesstream_6/concat:output:0^stream_6/ReadVariableOp*
_output_shapes
 *
dtype02
stream_6/AssignVariableOpø
'stream_6/conv2d_6/Conv2D/ReadVariableOpReadVariableOp@stream_6_conv2d_6_conv2d_readvariableop_stream_6_conv2d_6_kernel^stream_6/AssignVariableOp*'
_output_shapes
:@*
dtype02)
'stream_6/conv2d_6/Conv2D/ReadVariableOpä
stream_6/conv2d_6/Conv2DConv2Dstream_6/concat:output:0/stream_6/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:*
paddingVALID*
strides
2
stream_6/conv2d_6/Conv2Dí
(stream_6/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp?stream_6_conv2d_6_biasadd_readvariableop_stream_6_conv2d_6_bias^stream_6/AssignVariableOp*
_output_shapes	
:*
dtype02*
(stream_6/conv2d_6/BiasAdd/ReadVariableOpÈ
stream_6/conv2d_6/BiasAddBiasAdd!stream_6/conv2d_6/Conv2D:output:00stream_6/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:2
stream_6/conv2d_6/BiasAdd
stream_6/conv2d_6/ReluRelu"stream_6/conv2d_6/BiasAdd:output:0*
T0*'
_output_shapes
:2
stream_6/conv2d_6/Relu­
stream_7/ReadVariableOpReadVariableOp1stream_7_readvariableop_streaming_stream_7_states*'
_output_shapes
:*
dtype02
stream_7/ReadVariableOp
stream_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
stream_7/strided_slice/stack
stream_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2 
stream_7/strided_slice/stack_1
stream_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2 
stream_7/strided_slice/stack_2¹
stream_7/strided_sliceStridedSlicestream_7/ReadVariableOp:value:0%stream_7/strided_slice/stack:output:0'stream_7/strided_slice/stack_1:output:0'stream_7/strided_slice/stack_2:output:0*
Index0*
T0*%
_output_shapes
: *

begin_mask*
end_mask2
stream_7/strided_slicen
stream_7/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
stream_7/concat/axisÏ
stream_7/concatConcatV2stream_7/strided_slice:output:0$stream_6/conv2d_6/Relu:activations:0stream_7/concat/axis:output:0*
N*
T0*'
_output_shapes
:2
stream_7/concatÔ
stream_7/AssignVariableOpAssignVariableOp1stream_7_readvariableop_streaming_stream_7_statesstream_7/concat:output:0^stream_7/ReadVariableOp*
_output_shapes
 *
dtype02
stream_7/AssignVariableOp
stream_7/flatten/ConstConst^stream_7/AssignVariableOp*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
stream_7/flatten/Const¤
stream_7/flatten/ReshapeReshapestream_7/concat:output:0stream_7/flatten/Const:output:0*
T0*
_output_shapes
:	2
stream_7/flatten/Reshapes
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/dropout/Const
dropout/dropout/MulMul!stream_7/flatten/Reshape:output:0dropout/dropout/Const:output:0*
T0*
_output_shapes
:	2
dropout/dropout/Mul
dropout/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     2
dropout/dropout/ShapeÄ
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*
_output_shapes
:	*
dtype02.
,dropout/dropout/random_uniform/RandomUniform
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2 
dropout/dropout/GreaterEqual/yÖ
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	2
dropout/dropout/GreaterEqual
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	2
dropout/dropout/Cast
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*
_output_shapes
:	2
dropout/dropout/Mul_1¯
dense/MatMul/ReadVariableOpReadVariableOp2dense_matmul_readvariableop_streaming_dense_kernel* 
_output_shapes
:
*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMuldropout/dropout/Mul_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense/MatMul«
dense/BiasAdd/ReadVariableOpReadVariableOp1dense_biasadd_readvariableop_streaming_dense_bias*
_output_shapes	
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense/BiasAdd·
dense_1/MatMul/ReadVariableOpReadVariableOp6dense_1_matmul_readvariableop_streaming_dense_1_kernel* 
_output_shapes
:
*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldense/BiasAdd:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense_1/MatMul³
dense_1/BiasAdd/ReadVariableOpReadVariableOp5dense_1_biasadd_readvariableop_streaming_dense_1_bias*
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
dense_1/Relu¶
dense_2/MatMul/ReadVariableOpReadVariableOp6dense_2_matmul_readvariableop_streaming_dense_2_kernel*
_output_shapes
:	*
dtype02
dense_2/MatMul/ReadVariableOp
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_2/MatMul²
dense_2/BiasAdd/ReadVariableOpReadVariableOp5dense_2_biasadd_readvariableop_streaming_dense_2_bias*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_2/BiasAddÁ
IdentityIdentitydense_2/BiasAdd:output:0^stream/AssignVariableOp^stream_1/AssignVariableOp^stream_2/AssignVariableOp^stream_3/AssignVariableOp^stream_4/AssignVariableOp^stream_5/AssignVariableOp^stream_6/AssignVariableOp^stream_7/AssignVariableOp*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*
_input_shapes
~:::::::::::::::::::::::::::::22
stream/AssignVariableOpstream/AssignVariableOp26
stream_1/AssignVariableOpstream_1/AssignVariableOp26
stream_2/AssignVariableOpstream_2/AssignVariableOp26
stream_3/AssignVariableOpstream_3/AssignVariableOp26
stream_4/AssignVariableOpstream_4/AssignVariableOp26
stream_5/AssignVariableOpstream_5/AssignVariableOp26
stream_6/AssignVariableOpstream_6/AssignVariableOp26
stream_7/AssignVariableOpstream_7/AssignVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ì
¾
@__inference_conv2d_layer_call_and_return_conditional_losses_2590

inputs.
*conv2d_readvariableop_stream_conv2d_kernel-
)biasadd_readvariableop_stream_conv2d_bias
identityo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate¡
Conv2D/ReadVariableOpReadVariableOp*conv2d_readvariableop_stream_conv2d_kernel*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp¶
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOp)biasadd_readvariableop_stream_conv2d_bias*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2
Relu
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Þ
È
B__inference_conv2d_1_layer_call_and_return_conditional_losses_2638

inputs2
.conv2d_readvariableop_stream_1_conv2d_1_kernel1
-biasadd_readvariableop_stream_1_conv2d_1_bias
identityo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate¥
Conv2D/ReadVariableOpReadVariableOp.conv2d_readvariableop_stream_1_conv2d_1_kernel*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¶
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOp-biasadd_readvariableop_stream_1_conv2d_1_bias*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2
Relu
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
é
B
&__inference_dropout_layer_call_fn_4380

inputs
identity´
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_31052
PartitionedCalld
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs


B__inference_stream_6_layer_call_and_return_conditional_losses_4325

inputs,
(readvariableop_streaming_stream_6_states;
7conv2d_6_conv2d_readvariableop_stream_6_conv2d_6_kernel:
6conv2d_6_biasadd_readvariableop_stream_6_conv2d_6_bias
identity¢AssignVariableOp
ReadVariableOpReadVariableOp(readvariableop_streaming_stream_6_states*&
_output_shapes
:@*
dtype02
ReadVariableOp
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:@*

begin_mask*
end_mask2
strided_slice\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2strided_slice:output:0inputsconcat/axis:output:0*
N*
T0*&
_output_shapes
:@2
concat§
AssignVariableOpAssignVariableOp(readvariableop_streaming_stream_6_statesconcat:output:0^ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOpÔ
conv2d_6/Conv2D/ReadVariableOpReadVariableOp7conv2d_6_conv2d_readvariableop_stream_6_conv2d_6_kernel^AssignVariableOp*'
_output_shapes
:@*
dtype02 
conv2d_6/Conv2D/ReadVariableOpÀ
conv2d_6/Conv2DConv2Dconcat:output:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:*
paddingVALID*
strides
2
conv2d_6/Conv2DÉ
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp6conv2d_6_biasadd_readvariableop_stream_6_conv2d_6_bias^AssignVariableOp*
_output_shapes	
:*
dtype02!
conv2d_6/BiasAdd/ReadVariableOp¤
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:2
conv2d_6/BiasAdds
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*'
_output_shapes
:2
conv2d_6/Relu
IdentityIdentityconv2d_6/Relu:activations:0^AssignVariableOp*
T0*'
_output_shapes
:2

Identity"
identityIdentity:output:0*1
_input_shapes 
:@:::2$
AssignVariableOpAssignVariableOp:N J
&
_output_shapes
:@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
¶

'__inference_conv2d_3_layer_call_fn_2709

inputs
stream_3_conv2d_3_kernel
stream_3_conv2d_3_bias
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsstream_3_conv2d_3_kernelstream_3_conv2d_3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_27042
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 

Ç
A__inference_dense_2_layer_call_and_return_conditional_losses_4425

inputs2
.matmul_readvariableop_streaming_dense_2_kernel1
-biasadd_readvariableop_streaming_dense_2_bias
identity
MatMul/ReadVariableOpReadVariableOp.matmul_readvariableop_streaming_dense_2_kernel*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
MatMul
BiasAdd/ReadVariableOpReadVariableOp-biasadd_readvariableop_streaming_dense_2_bias*
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
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
©
¹
'__inference_stream_5_layer_call_fn_4306

inputs
streaming_stream_5_states
stream_5_conv2d_5_kernel
stream_5_conv2d_5_bias
identity¢StatefulPartitionedCall¨
StatefulPartitionedCallStatefulPartitionedCallinputsstreaming_stream_5_statesstream_5_conv2d_5_kernelstream_5_conv2d_5_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_stream_5_layer_call_and_return_conditional_losses_30162
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*2
_input_shapes!
::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
øH
¢
__inference__traced_save_4543
file_prefix6
2savev2_streaming_stream_states_read_readvariableop8
4savev2_streaming_stream_1_states_read_readvariableop8
4savev2_streaming_stream_2_states_read_readvariableop8
4savev2_streaming_stream_3_states_read_readvariableop8
4savev2_streaming_stream_4_states_read_readvariableop8
4savev2_streaming_stream_5_states_read_readvariableop8
4savev2_streaming_stream_6_states_read_readvariableop8
4savev2_streaming_stream_7_states_read_readvariableop5
1savev2_streaming_dense_kernel_read_readvariableop3
/savev2_streaming_dense_bias_read_readvariableop7
3savev2_streaming_dense_1_kernel_read_readvariableop5
1savev2_streaming_dense_1_bias_read_readvariableop7
3savev2_streaming_dense_2_kernel_read_readvariableop5
1savev2_streaming_dense_2_bias_read_readvariableop3
/savev2_stream_conv2d_kernel_read_readvariableop1
-savev2_stream_conv2d_bias_read_readvariableop7
3savev2_stream_1_conv2d_1_kernel_read_readvariableop5
1savev2_stream_1_conv2d_1_bias_read_readvariableop7
3savev2_stream_2_conv2d_2_kernel_read_readvariableop5
1savev2_stream_2_conv2d_2_bias_read_readvariableop7
3savev2_stream_3_conv2d_3_kernel_read_readvariableop5
1savev2_stream_3_conv2d_3_bias_read_readvariableop7
3savev2_stream_4_conv2d_4_kernel_read_readvariableop5
1savev2_stream_4_conv2d_4_bias_read_readvariableop7
3savev2_stream_5_conv2d_5_kernel_read_readvariableop5
1savev2_stream_5_conv2d_5_bias_read_readvariableop7
3savev2_stream_6_conv2d_6_kernel_read_readvariableop5
1savev2_stream_6_conv2d_6_bias_read_readvariableop
savev2_1_const

identity_1¢MergeV2Checkpoints¢SaveV2¢SaveV2_1
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
value3B1 B+_temp_02f5435a958e449a8a275c9e8e4b8e82/part2	
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
value	B :2

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
ShardedFilenameÆ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ø

valueÎ
BË
B6layer_with_weights-0/states/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/states/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/states/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/states/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/states/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/states/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/states/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/states/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_namesÀ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesù
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:02savev2_streaming_stream_states_read_readvariableop4savev2_streaming_stream_1_states_read_readvariableop4savev2_streaming_stream_2_states_read_readvariableop4savev2_streaming_stream_3_states_read_readvariableop4savev2_streaming_stream_4_states_read_readvariableop4savev2_streaming_stream_5_states_read_readvariableop4savev2_streaming_stream_6_states_read_readvariableop4savev2_streaming_stream_7_states_read_readvariableop1savev2_streaming_dense_kernel_read_readvariableop/savev2_streaming_dense_bias_read_readvariableop3savev2_streaming_dense_1_kernel_read_readvariableop1savev2_streaming_dense_1_bias_read_readvariableop3savev2_streaming_dense_2_kernel_read_readvariableop1savev2_streaming_dense_2_bias_read_readvariableop/savev2_stream_conv2d_kernel_read_readvariableop-savev2_stream_conv2d_bias_read_readvariableop3savev2_stream_1_conv2d_1_kernel_read_readvariableop1savev2_stream_1_conv2d_1_bias_read_readvariableop3savev2_stream_2_conv2d_2_kernel_read_readvariableop1savev2_stream_2_conv2d_2_bias_read_readvariableop3savev2_stream_3_conv2d_3_kernel_read_readvariableop1savev2_stream_3_conv2d_3_bias_read_readvariableop3savev2_stream_4_conv2d_4_kernel_read_readvariableop1savev2_stream_4_conv2d_4_bias_read_readvariableop3savev2_stream_5_conv2d_5_kernel_read_readvariableop1savev2_stream_5_conv2d_5_bias_read_readvariableop3savev2_stream_6_conv2d_6_kernel_read_readvariableop1savev2_stream_6_conv2d_6_bias_read_readvariableop"/device:CPU:0*
_output_shapes
 **
dtypes 
22
SaveV2
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard¬
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1¢
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slicesÏ
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1ã
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¬
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapesý
ú: ::@:@:@:@::@::
::
::	::@:@:@@:@:@@:@:@@:@:@::@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
::,(
&
_output_shapes
:@:,(
&
_output_shapes
:@:,(
&
_output_shapes
:@:,(
&
_output_shapes
:@:-)
'
_output_shapes
::,(
&
_output_shapes
:@:-)
'
_output_shapes
::&	"
 
_output_shapes
:
:!


_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:-)
'
_output_shapes
:@:!

_output_shapes	
::-)
'
_output_shapes
:@: 

_output_shapes
:@:-)
'
_output_shapes
:@:!

_output_shapes	
::

_output_shapes
: 
ä
È
B__inference_conv2d_6_layer_call_and_return_conditional_losses_2803

inputs2
.conv2d_readvariableop_stream_6_conv2d_6_kernel1
-biasadd_readvariableop_stream_6_conv2d_6_bias
identityo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate¦
Conv2D/ReadVariableOpReadVariableOp.conv2d_readvariableop_stream_6_conv2d_6_kernel*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp·
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOp-biasadd_readvariableop_stream_6_conv2d_6_bias*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
©
¹
'__inference_stream_4_layer_call_fn_4279

inputs
streaming_stream_4_states
stream_4_conv2d_4_kernel
stream_4_conv2d_4_bias
identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsstreaming_stream_4_statesstream_4_conv2d_4_kernelstream_4_conv2d_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_stream_4_layer_call_and_return_conditional_losses_29822
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:2

Identity"
identityIdentity:output:0*1
_input_shapes 
:@:::22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
«Ô

F__inference_functional_1_layer_call_and_return_conditional_losses_3926
input_audio1
-stream_readvariableop_streaming_stream_states<
8stream_conv2d_conv2d_readvariableop_stream_conv2d_kernel;
7stream_conv2d_biasadd_readvariableop_stream_conv2d_bias5
1stream_1_readvariableop_streaming_stream_1_statesD
@stream_1_conv2d_1_conv2d_readvariableop_stream_1_conv2d_1_kernelC
?stream_1_conv2d_1_biasadd_readvariableop_stream_1_conv2d_1_bias5
1stream_2_readvariableop_streaming_stream_2_statesD
@stream_2_conv2d_2_conv2d_readvariableop_stream_2_conv2d_2_kernelC
?stream_2_conv2d_2_biasadd_readvariableop_stream_2_conv2d_2_bias5
1stream_3_readvariableop_streaming_stream_3_statesD
@stream_3_conv2d_3_conv2d_readvariableop_stream_3_conv2d_3_kernelC
?stream_3_conv2d_3_biasadd_readvariableop_stream_3_conv2d_3_bias5
1stream_4_readvariableop_streaming_stream_4_statesD
@stream_4_conv2d_4_conv2d_readvariableop_stream_4_conv2d_4_kernelC
?stream_4_conv2d_4_biasadd_readvariableop_stream_4_conv2d_4_bias5
1stream_5_readvariableop_streaming_stream_5_statesD
@stream_5_conv2d_5_conv2d_readvariableop_stream_5_conv2d_5_kernelC
?stream_5_conv2d_5_biasadd_readvariableop_stream_5_conv2d_5_bias5
1stream_6_readvariableop_streaming_stream_6_statesD
@stream_6_conv2d_6_conv2d_readvariableop_stream_6_conv2d_6_kernelC
?stream_6_conv2d_6_biasadd_readvariableop_stream_6_conv2d_6_bias5
1stream_7_readvariableop_streaming_stream_7_states6
2dense_matmul_readvariableop_streaming_dense_kernel5
1dense_biasadd_readvariableop_streaming_dense_bias:
6dense_1_matmul_readvariableop_streaming_dense_1_kernel9
5dense_1_biasadd_readvariableop_streaming_dense_1_bias:
6dense_2_matmul_readvariableop_streaming_dense_2_kernel9
5dense_2_biasadd_readvariableop_streaming_dense_2_bias
identity¢stream/AssignVariableOp¢stream_1/AssignVariableOp¢stream_2/AssignVariableOp¢stream_3/AssignVariableOp¢stream_4/AssignVariableOp¢stream_5/AssignVariableOp¢stream_6/AssignVariableOp¢stream_7/AssignVariableOp
%tf_op_layer_ExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2'
%tf_op_layer_ExpandDims/ExpandDims/dimÑ
!tf_op_layer_ExpandDims/ExpandDims
ExpandDimsinput_audio.tf_op_layer_ExpandDims/ExpandDims/dim:output:0*
T0*
_cloned(*&
_output_shapes
:2#
!tf_op_layer_ExpandDims/ExpandDims¤
stream/ReadVariableOpReadVariableOp-stream_readvariableop_streaming_stream_states*&
_output_shapes
:*
dtype02
stream/ReadVariableOp
stream/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
stream/strided_slice/stack
stream/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
stream/strided_slice/stack_1
stream/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
stream/strided_slice/stack_2®
stream/strided_sliceStridedSlicestream/ReadVariableOp:value:0#stream/strided_slice/stack:output:0%stream/strided_slice/stack_1:output:0%stream/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:*

begin_mask*
end_mask2
stream/strided_slicej
stream/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
stream/concat/axisÌ
stream/concatConcatV2stream/strided_slice:output:0*tf_op_layer_ExpandDims/ExpandDims:output:0stream/concat/axis:output:0*
N*
T0*&
_output_shapes
:2
stream/concatÈ
stream/AssignVariableOpAssignVariableOp-stream_readvariableop_streaming_stream_statesstream/concat:output:0^stream/ReadVariableOp*
_output_shapes
 *
dtype02
stream/AssignVariableOpå
#stream/conv2d/Conv2D/ReadVariableOpReadVariableOp8stream_conv2d_conv2d_readvariableop_stream_conv2d_kernel^stream/AssignVariableOp*&
_output_shapes
:@*
dtype02%
#stream/conv2d/Conv2D/ReadVariableOpÕ
stream/conv2d/Conv2DConv2Dstream/concat:output:0+stream/conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
stream/conv2d/Conv2DÚ
$stream/conv2d/BiasAdd/ReadVariableOpReadVariableOp7stream_conv2d_biasadd_readvariableop_stream_conv2d_bias^stream/AssignVariableOp*
_output_shapes
:@*
dtype02&
$stream/conv2d/BiasAdd/ReadVariableOp·
stream/conv2d/BiasAddBiasAddstream/conv2d/Conv2D:output:0,stream/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
stream/conv2d/BiasAdd
stream/conv2d/ReluRelustream/conv2d/BiasAdd:output:0*
T0*&
_output_shapes
:@2
stream/conv2d/Relu¬
stream_1/ReadVariableOpReadVariableOp1stream_1_readvariableop_streaming_stream_1_states*&
_output_shapes
:@*
dtype02
stream_1/ReadVariableOp
stream_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
stream_1/strided_slice/stack
stream_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2 
stream_1/strided_slice/stack_1
stream_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2 
stream_1/strided_slice/stack_2º
stream_1/strided_sliceStridedSlicestream_1/ReadVariableOp:value:0%stream_1/strided_slice/stack:output:0'stream_1/strided_slice/stack_1:output:0'stream_1/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:@*

begin_mask*
end_mask2
stream_1/strided_slicen
stream_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
stream_1/concat/axisÊ
stream_1/concatConcatV2stream_1/strided_slice:output:0 stream/conv2d/Relu:activations:0stream_1/concat/axis:output:0*
N*
T0*&
_output_shapes
:@2
stream_1/concatÔ
stream_1/AssignVariableOpAssignVariableOp1stream_1_readvariableop_streaming_stream_1_statesstream_1/concat:output:0^stream_1/ReadVariableOp*
_output_shapes
 *
dtype02
stream_1/AssignVariableOp÷
'stream_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp@stream_1_conv2d_1_conv2d_readvariableop_stream_1_conv2d_1_kernel^stream_1/AssignVariableOp*&
_output_shapes
:@@*
dtype02)
'stream_1/conv2d_1/Conv2D/ReadVariableOpã
stream_1/conv2d_1/Conv2DConv2Dstream_1/concat:output:0/stream_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
stream_1/conv2d_1/Conv2Dì
(stream_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp?stream_1_conv2d_1_biasadd_readvariableop_stream_1_conv2d_1_bias^stream_1/AssignVariableOp*
_output_shapes
:@*
dtype02*
(stream_1/conv2d_1/BiasAdd/ReadVariableOpÇ
stream_1/conv2d_1/BiasAddBiasAdd!stream_1/conv2d_1/Conv2D:output:00stream_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
stream_1/conv2d_1/BiasAdd
stream_1/conv2d_1/ReluRelu"stream_1/conv2d_1/BiasAdd:output:0*
T0*&
_output_shapes
:@2
stream_1/conv2d_1/Relu¬
stream_2/ReadVariableOpReadVariableOp1stream_2_readvariableop_streaming_stream_2_states*&
_output_shapes
:@*
dtype02
stream_2/ReadVariableOp
stream_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
stream_2/strided_slice/stack
stream_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2 
stream_2/strided_slice/stack_1
stream_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2 
stream_2/strided_slice/stack_2º
stream_2/strided_sliceStridedSlicestream_2/ReadVariableOp:value:0%stream_2/strided_slice/stack:output:0'stream_2/strided_slice/stack_1:output:0'stream_2/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:@*

begin_mask*
end_mask2
stream_2/strided_slicen
stream_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
stream_2/concat/axisÎ
stream_2/concatConcatV2stream_2/strided_slice:output:0$stream_1/conv2d_1/Relu:activations:0stream_2/concat/axis:output:0*
N*
T0*&
_output_shapes
:@2
stream_2/concatÔ
stream_2/AssignVariableOpAssignVariableOp1stream_2_readvariableop_streaming_stream_2_statesstream_2/concat:output:0^stream_2/ReadVariableOp*
_output_shapes
 *
dtype02
stream_2/AssignVariableOp÷
'stream_2/conv2d_2/Conv2D/ReadVariableOpReadVariableOp@stream_2_conv2d_2_conv2d_readvariableop_stream_2_conv2d_2_kernel^stream_2/AssignVariableOp*&
_output_shapes
:@@*
dtype02)
'stream_2/conv2d_2/Conv2D/ReadVariableOpã
stream_2/conv2d_2/Conv2DConv2Dstream_2/concat:output:0/stream_2/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
stream_2/conv2d_2/Conv2Dì
(stream_2/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp?stream_2_conv2d_2_biasadd_readvariableop_stream_2_conv2d_2_bias^stream_2/AssignVariableOp*
_output_shapes
:@*
dtype02*
(stream_2/conv2d_2/BiasAdd/ReadVariableOpÇ
stream_2/conv2d_2/BiasAddBiasAdd!stream_2/conv2d_2/Conv2D:output:00stream_2/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
stream_2/conv2d_2/BiasAdd
stream_2/conv2d_2/ReluRelu"stream_2/conv2d_2/BiasAdd:output:0*
T0*&
_output_shapes
:@2
stream_2/conv2d_2/Relu¬
stream_3/ReadVariableOpReadVariableOp1stream_3_readvariableop_streaming_stream_3_states*&
_output_shapes
:@*
dtype02
stream_3/ReadVariableOp
stream_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
stream_3/strided_slice/stack
stream_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2 
stream_3/strided_slice/stack_1
stream_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2 
stream_3/strided_slice/stack_2º
stream_3/strided_sliceStridedSlicestream_3/ReadVariableOp:value:0%stream_3/strided_slice/stack:output:0'stream_3/strided_slice/stack_1:output:0'stream_3/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:@*

begin_mask*
end_mask2
stream_3/strided_slicen
stream_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
stream_3/concat/axisÎ
stream_3/concatConcatV2stream_3/strided_slice:output:0$stream_2/conv2d_2/Relu:activations:0stream_3/concat/axis:output:0*
N*
T0*&
_output_shapes
:@2
stream_3/concatÔ
stream_3/AssignVariableOpAssignVariableOp1stream_3_readvariableop_streaming_stream_3_statesstream_3/concat:output:0^stream_3/ReadVariableOp*
_output_shapes
 *
dtype02
stream_3/AssignVariableOp÷
'stream_3/conv2d_3/Conv2D/ReadVariableOpReadVariableOp@stream_3_conv2d_3_conv2d_readvariableop_stream_3_conv2d_3_kernel^stream_3/AssignVariableOp*&
_output_shapes
:@@*
dtype02)
'stream_3/conv2d_3/Conv2D/ReadVariableOpã
stream_3/conv2d_3/Conv2DConv2Dstream_3/concat:output:0/stream_3/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
stream_3/conv2d_3/Conv2Dì
(stream_3/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp?stream_3_conv2d_3_biasadd_readvariableop_stream_3_conv2d_3_bias^stream_3/AssignVariableOp*
_output_shapes
:@*
dtype02*
(stream_3/conv2d_3/BiasAdd/ReadVariableOpÇ
stream_3/conv2d_3/BiasAddBiasAdd!stream_3/conv2d_3/Conv2D:output:00stream_3/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
stream_3/conv2d_3/BiasAdd
stream_3/conv2d_3/ReluRelu"stream_3/conv2d_3/BiasAdd:output:0*
T0*&
_output_shapes
:@2
stream_3/conv2d_3/Relu¬
stream_4/ReadVariableOpReadVariableOp1stream_4_readvariableop_streaming_stream_4_states*&
_output_shapes
:@*
dtype02
stream_4/ReadVariableOp
stream_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
stream_4/strided_slice/stack
stream_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2 
stream_4/strided_slice/stack_1
stream_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2 
stream_4/strided_slice/stack_2º
stream_4/strided_sliceStridedSlicestream_4/ReadVariableOp:value:0%stream_4/strided_slice/stack:output:0'stream_4/strided_slice/stack_1:output:0'stream_4/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:@*

begin_mask*
end_mask2
stream_4/strided_slicen
stream_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
stream_4/concat/axisÎ
stream_4/concatConcatV2stream_4/strided_slice:output:0$stream_3/conv2d_3/Relu:activations:0stream_4/concat/axis:output:0*
N*
T0*&
_output_shapes
:@2
stream_4/concatÔ
stream_4/AssignVariableOpAssignVariableOp1stream_4_readvariableop_streaming_stream_4_statesstream_4/concat:output:0^stream_4/ReadVariableOp*
_output_shapes
 *
dtype02
stream_4/AssignVariableOpø
'stream_4/conv2d_4/Conv2D/ReadVariableOpReadVariableOp@stream_4_conv2d_4_conv2d_readvariableop_stream_4_conv2d_4_kernel^stream_4/AssignVariableOp*'
_output_shapes
:@*
dtype02)
'stream_4/conv2d_4/Conv2D/ReadVariableOpä
stream_4/conv2d_4/Conv2DConv2Dstream_4/concat:output:0/stream_4/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:*
paddingVALID*
strides
2
stream_4/conv2d_4/Conv2Dí
(stream_4/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp?stream_4_conv2d_4_biasadd_readvariableop_stream_4_conv2d_4_bias^stream_4/AssignVariableOp*
_output_shapes	
:*
dtype02*
(stream_4/conv2d_4/BiasAdd/ReadVariableOpÈ
stream_4/conv2d_4/BiasAddBiasAdd!stream_4/conv2d_4/Conv2D:output:00stream_4/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:2
stream_4/conv2d_4/BiasAdd
stream_4/conv2d_4/ReluRelu"stream_4/conv2d_4/BiasAdd:output:0*
T0*'
_output_shapes
:2
stream_4/conv2d_4/Relu­
stream_5/ReadVariableOpReadVariableOp1stream_5_readvariableop_streaming_stream_5_states*'
_output_shapes
:*
dtype02
stream_5/ReadVariableOp
stream_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
stream_5/strided_slice/stack
stream_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2 
stream_5/strided_slice/stack_1
stream_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2 
stream_5/strided_slice/stack_2»
stream_5/strided_sliceStridedSlicestream_5/ReadVariableOp:value:0%stream_5/strided_slice/stack:output:0'stream_5/strided_slice/stack_1:output:0'stream_5/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:*

begin_mask*
end_mask2
stream_5/strided_slicen
stream_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
stream_5/concat/axisÏ
stream_5/concatConcatV2stream_5/strided_slice:output:0$stream_4/conv2d_4/Relu:activations:0stream_5/concat/axis:output:0*
N*
T0*'
_output_shapes
:2
stream_5/concatÔ
stream_5/AssignVariableOpAssignVariableOp1stream_5_readvariableop_streaming_stream_5_statesstream_5/concat:output:0^stream_5/ReadVariableOp*
_output_shapes
 *
dtype02
stream_5/AssignVariableOpø
'stream_5/conv2d_5/Conv2D/ReadVariableOpReadVariableOp@stream_5_conv2d_5_conv2d_readvariableop_stream_5_conv2d_5_kernel^stream_5/AssignVariableOp*'
_output_shapes
:@*
dtype02)
'stream_5/conv2d_5/Conv2D/ReadVariableOpã
stream_5/conv2d_5/Conv2DConv2Dstream_5/concat:output:0/stream_5/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
stream_5/conv2d_5/Conv2Dì
(stream_5/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp?stream_5_conv2d_5_biasadd_readvariableop_stream_5_conv2d_5_bias^stream_5/AssignVariableOp*
_output_shapes
:@*
dtype02*
(stream_5/conv2d_5/BiasAdd/ReadVariableOpÇ
stream_5/conv2d_5/BiasAddBiasAdd!stream_5/conv2d_5/Conv2D:output:00stream_5/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
stream_5/conv2d_5/BiasAdd
stream_5/conv2d_5/ReluRelu"stream_5/conv2d_5/BiasAdd:output:0*
T0*&
_output_shapes
:@2
stream_5/conv2d_5/Relu¬
stream_6/ReadVariableOpReadVariableOp1stream_6_readvariableop_streaming_stream_6_states*&
_output_shapes
:@*
dtype02
stream_6/ReadVariableOp
stream_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
stream_6/strided_slice/stack
stream_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2 
stream_6/strided_slice/stack_1
stream_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2 
stream_6/strided_slice/stack_2º
stream_6/strided_sliceStridedSlicestream_6/ReadVariableOp:value:0%stream_6/strided_slice/stack:output:0'stream_6/strided_slice/stack_1:output:0'stream_6/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:@*

begin_mask*
end_mask2
stream_6/strided_slicen
stream_6/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
stream_6/concat/axisÎ
stream_6/concatConcatV2stream_6/strided_slice:output:0$stream_5/conv2d_5/Relu:activations:0stream_6/concat/axis:output:0*
N*
T0*&
_output_shapes
:@2
stream_6/concatÔ
stream_6/AssignVariableOpAssignVariableOp1stream_6_readvariableop_streaming_stream_6_statesstream_6/concat:output:0^stream_6/ReadVariableOp*
_output_shapes
 *
dtype02
stream_6/AssignVariableOpø
'stream_6/conv2d_6/Conv2D/ReadVariableOpReadVariableOp@stream_6_conv2d_6_conv2d_readvariableop_stream_6_conv2d_6_kernel^stream_6/AssignVariableOp*'
_output_shapes
:@*
dtype02)
'stream_6/conv2d_6/Conv2D/ReadVariableOpä
stream_6/conv2d_6/Conv2DConv2Dstream_6/concat:output:0/stream_6/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:*
paddingVALID*
strides
2
stream_6/conv2d_6/Conv2Dí
(stream_6/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp?stream_6_conv2d_6_biasadd_readvariableop_stream_6_conv2d_6_bias^stream_6/AssignVariableOp*
_output_shapes	
:*
dtype02*
(stream_6/conv2d_6/BiasAdd/ReadVariableOpÈ
stream_6/conv2d_6/BiasAddBiasAdd!stream_6/conv2d_6/Conv2D:output:00stream_6/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:2
stream_6/conv2d_6/BiasAdd
stream_6/conv2d_6/ReluRelu"stream_6/conv2d_6/BiasAdd:output:0*
T0*'
_output_shapes
:2
stream_6/conv2d_6/Relu­
stream_7/ReadVariableOpReadVariableOp1stream_7_readvariableop_streaming_stream_7_states*'
_output_shapes
:*
dtype02
stream_7/ReadVariableOp
stream_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
stream_7/strided_slice/stack
stream_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2 
stream_7/strided_slice/stack_1
stream_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2 
stream_7/strided_slice/stack_2¹
stream_7/strided_sliceStridedSlicestream_7/ReadVariableOp:value:0%stream_7/strided_slice/stack:output:0'stream_7/strided_slice/stack_1:output:0'stream_7/strided_slice/stack_2:output:0*
Index0*
T0*%
_output_shapes
: *

begin_mask*
end_mask2
stream_7/strided_slicen
stream_7/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
stream_7/concat/axisÏ
stream_7/concatConcatV2stream_7/strided_slice:output:0$stream_6/conv2d_6/Relu:activations:0stream_7/concat/axis:output:0*
N*
T0*'
_output_shapes
:2
stream_7/concatÔ
stream_7/AssignVariableOpAssignVariableOp1stream_7_readvariableop_streaming_stream_7_statesstream_7/concat:output:0^stream_7/ReadVariableOp*
_output_shapes
 *
dtype02
stream_7/AssignVariableOp
stream_7/flatten/ConstConst^stream_7/AssignVariableOp*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
stream_7/flatten/Const¤
stream_7/flatten/ReshapeReshapestream_7/concat:output:0stream_7/flatten/Const:output:0*
T0*
_output_shapes
:	2
stream_7/flatten/Reshapes
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/dropout/Const
dropout/dropout/MulMul!stream_7/flatten/Reshape:output:0dropout/dropout/Const:output:0*
T0*
_output_shapes
:	2
dropout/dropout/Mul
dropout/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     2
dropout/dropout/ShapeÄ
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*
_output_shapes
:	*
dtype02.
,dropout/dropout/random_uniform/RandomUniform
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2 
dropout/dropout/GreaterEqual/yÖ
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	2
dropout/dropout/GreaterEqual
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	2
dropout/dropout/Cast
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*
_output_shapes
:	2
dropout/dropout/Mul_1¯
dense/MatMul/ReadVariableOpReadVariableOp2dense_matmul_readvariableop_streaming_dense_kernel* 
_output_shapes
:
*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMuldropout/dropout/Mul_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense/MatMul«
dense/BiasAdd/ReadVariableOpReadVariableOp1dense_biasadd_readvariableop_streaming_dense_bias*
_output_shapes	
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense/BiasAdd·
dense_1/MatMul/ReadVariableOpReadVariableOp6dense_1_matmul_readvariableop_streaming_dense_1_kernel* 
_output_shapes
:
*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldense/BiasAdd:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense_1/MatMul³
dense_1/BiasAdd/ReadVariableOpReadVariableOp5dense_1_biasadd_readvariableop_streaming_dense_1_bias*
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
dense_1/Relu¶
dense_2/MatMul/ReadVariableOpReadVariableOp6dense_2_matmul_readvariableop_streaming_dense_2_kernel*
_output_shapes
:	*
dtype02
dense_2/MatMul/ReadVariableOp
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_2/MatMul²
dense_2/BiasAdd/ReadVariableOpReadVariableOp5dense_2_biasadd_readvariableop_streaming_dense_2_bias*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_2/BiasAddÁ
IdentityIdentitydense_2/BiasAdd:output:0^stream/AssignVariableOp^stream_1/AssignVariableOp^stream_2/AssignVariableOp^stream_3/AssignVariableOp^stream_4/AssignVariableOp^stream_5/AssignVariableOp^stream_6/AssignVariableOp^stream_7/AssignVariableOp*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*
_input_shapes
~:::::::::::::::::::::::::::::22
stream/AssignVariableOpstream/AssignVariableOp26
stream_1/AssignVariableOpstream_1/AssignVariableOp26
stream_2/AssignVariableOpstream_2/AssignVariableOp26
stream_3/AssignVariableOpstream_3/AssignVariableOp26
stream_4/AssignVariableOpstream_4/AssignVariableOp26
stream_5/AssignVariableOpstream_5/AssignVariableOp26
stream_6/AssignVariableOpstream_6/AssignVariableOp26
stream_7/AssignVariableOpstream_7/AssignVariableOp:X T
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinput_audio:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Æ
¢
B__inference_stream_7_layer_call_and_return_conditional_losses_4347

inputs,
(readvariableop_streaming_stream_7_states
identity¢AssignVariableOp
ReadVariableOpReadVariableOp(readvariableop_streaming_stream_7_states*'
_output_shapes
:*
dtype02
ReadVariableOp
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*%
_output_shapes
: *

begin_mask*
end_mask2
strided_slice\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2strided_slice:output:0inputsconcat/axis:output:0*
N*
T0*'
_output_shapes
:2
concat§
AssignVariableOpAssignVariableOp(readvariableop_streaming_stream_7_statesconcat:output:0^ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOp
flatten/ConstConst^AssignVariableOp*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
flatten/Const
flatten/ReshapeReshapeconcat:output:0flatten/Const:output:0*
T0*
_output_shapes
:	2
flatten/Reshapew
IdentityIdentityflatten/Reshape:output:0^AssignVariableOp*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0**
_input_shapes
::2$
AssignVariableOpAssignVariableOp:O K
'
_output_shapes
:
 
_user_specified_nameinputs:

_output_shapes
: 
ï

+__inference_functional_1_layer_call_fn_3745

inputs
streaming_stream_states
stream_conv2d_kernel
stream_conv2d_bias
streaming_stream_1_states
stream_1_conv2d_1_kernel
stream_1_conv2d_1_bias
streaming_stream_2_states
stream_2_conv2d_2_kernel
stream_2_conv2d_2_bias
streaming_stream_3_states
stream_3_conv2d_3_kernel
stream_3_conv2d_3_bias
streaming_stream_4_states
stream_4_conv2d_4_kernel
stream_4_conv2d_4_bias
streaming_stream_5_states
stream_5_conv2d_5_kernel
stream_5_conv2d_5_bias
streaming_stream_6_states
stream_6_conv2d_6_kernel
stream_6_conv2d_6_bias
streaming_stream_7_states
streaming_dense_kernel
streaming_dense_bias
streaming_dense_1_kernel
streaming_dense_1_bias
streaming_dense_2_kernel
streaming_dense_2_bias
identity¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCallinputsstreaming_stream_statesstream_conv2d_kernelstream_conv2d_biasstreaming_stream_1_statesstream_1_conv2d_1_kernelstream_1_conv2d_1_biasstreaming_stream_2_statesstream_2_conv2d_2_kernelstream_2_conv2d_2_biasstreaming_stream_3_statesstream_3_conv2d_3_kernelstream_3_conv2d_3_biasstreaming_stream_4_statesstream_4_conv2d_4_kernelstream_4_conv2d_4_biasstreaming_stream_5_statesstream_5_conv2d_5_kernelstream_5_conv2d_5_biasstreaming_stream_6_statesstream_6_conv2d_6_kernelstream_6_conv2d_6_biasstreaming_stream_7_statesstreaming_dense_kernelstreaming_dense_biasstreaming_dense_1_kernelstreaming_dense_1_biasstreaming_dense_2_kernelstreaming_dense_2_bias*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*6
_read_only_resource_inputs
	**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_functional_1_layer_call_and_return_conditional_losses_32792
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 


B__inference_stream_4_layer_call_and_return_conditional_losses_4271

inputs,
(readvariableop_streaming_stream_4_states;
7conv2d_4_conv2d_readvariableop_stream_4_conv2d_4_kernel:
6conv2d_4_biasadd_readvariableop_stream_4_conv2d_4_bias
identity¢AssignVariableOp
ReadVariableOpReadVariableOp(readvariableop_streaming_stream_4_states*&
_output_shapes
:@*
dtype02
ReadVariableOp
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:@*

begin_mask*
end_mask2
strided_slice\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2strided_slice:output:0inputsconcat/axis:output:0*
N*
T0*&
_output_shapes
:@2
concat§
AssignVariableOpAssignVariableOp(readvariableop_streaming_stream_4_statesconcat:output:0^ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOpÔ
conv2d_4/Conv2D/ReadVariableOpReadVariableOp7conv2d_4_conv2d_readvariableop_stream_4_conv2d_4_kernel^AssignVariableOp*'
_output_shapes
:@*
dtype02 
conv2d_4/Conv2D/ReadVariableOpÀ
conv2d_4/Conv2DConv2Dconcat:output:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:*
paddingVALID*
strides
2
conv2d_4/Conv2DÉ
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp6conv2d_4_biasadd_readvariableop_stream_4_conv2d_4_bias^AssignVariableOp*
_output_shapes	
:*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp¤
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:2
conv2d_4/BiasAdds
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*'
_output_shapes
:2
conv2d_4/Relu
IdentityIdentityconv2d_4/Relu:activations:0^AssignVariableOp*
T0*'
_output_shapes
:2

Identity"
identityIdentity:output:0*1
_input_shapes 
:@:::2$
AssignVariableOpAssignVariableOp:N J
&
_output_shapes
:@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Û
l
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_4139

inputs
identityk
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
ExpandDims/dim

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*
_cloned(*&
_output_shapes
:2

ExpandDimsf
IdentityIdentityExpandDims:output:0*
T0*&
_output_shapes
:2

Identity"
identityIdentity:output:0*!
_input_shapes
::J F
"
_output_shapes
:
 
_user_specified_nameinputs
þz

 __inference__traced_restore_4639
file_prefix,
(assignvariableop_streaming_stream_states0
,assignvariableop_1_streaming_stream_1_states0
,assignvariableop_2_streaming_stream_2_states0
,assignvariableop_3_streaming_stream_3_states0
,assignvariableop_4_streaming_stream_4_states0
,assignvariableop_5_streaming_stream_5_states0
,assignvariableop_6_streaming_stream_6_states0
,assignvariableop_7_streaming_stream_7_states-
)assignvariableop_8_streaming_dense_kernel+
'assignvariableop_9_streaming_dense_bias0
,assignvariableop_10_streaming_dense_1_kernel.
*assignvariableop_11_streaming_dense_1_bias0
,assignvariableop_12_streaming_dense_2_kernel.
*assignvariableop_13_streaming_dense_2_bias,
(assignvariableop_14_stream_conv2d_kernel*
&assignvariableop_15_stream_conv2d_bias0
,assignvariableop_16_stream_1_conv2d_1_kernel.
*assignvariableop_17_stream_1_conv2d_1_bias0
,assignvariableop_18_stream_2_conv2d_2_kernel.
*assignvariableop_19_stream_2_conv2d_2_bias0
,assignvariableop_20_stream_3_conv2d_3_kernel.
*assignvariableop_21_stream_3_conv2d_3_bias0
,assignvariableop_22_stream_4_conv2d_4_kernel.
*assignvariableop_23_stream_4_conv2d_4_bias0
,assignvariableop_24_stream_5_conv2d_5_kernel.
*assignvariableop_25_stream_5_conv2d_5_bias0
,assignvariableop_26_stream_6_conv2d_6_kernel.
*assignvariableop_27_stream_6_conv2d_6_bias
identity_29¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9¢	RestoreV2¢RestoreV2_1Ì
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ø

valueÎ
BË
B6layer_with_weights-0/states/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/states/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/states/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/states/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/states/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/states/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/states/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/states/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_namesÆ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices¸
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesr
p::::::::::::::::::::::::::::**
dtypes 
22
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp(assignvariableop_streaming_stream_statesIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1¢
AssignVariableOp_1AssignVariableOp,assignvariableop_1_streaming_stream_1_statesIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2¢
AssignVariableOp_2AssignVariableOp,assignvariableop_2_streaming_stream_2_statesIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3¢
AssignVariableOp_3AssignVariableOp,assignvariableop_3_streaming_stream_3_statesIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4¢
AssignVariableOp_4AssignVariableOp,assignvariableop_4_streaming_stream_4_statesIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5¢
AssignVariableOp_5AssignVariableOp,assignvariableop_5_streaming_stream_5_statesIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6¢
AssignVariableOp_6AssignVariableOp,assignvariableop_6_streaming_stream_6_statesIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7¢
AssignVariableOp_7AssignVariableOp,assignvariableop_7_streaming_stream_7_statesIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8
AssignVariableOp_8AssignVariableOp)assignvariableop_8_streaming_dense_kernelIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9
AssignVariableOp_9AssignVariableOp'assignvariableop_9_streaming_dense_biasIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10¥
AssignVariableOp_10AssignVariableOp,assignvariableop_10_streaming_dense_1_kernelIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11£
AssignVariableOp_11AssignVariableOp*assignvariableop_11_streaming_dense_1_biasIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12¥
AssignVariableOp_12AssignVariableOp,assignvariableop_12_streaming_dense_2_kernelIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13£
AssignVariableOp_13AssignVariableOp*assignvariableop_13_streaming_dense_2_biasIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14¡
AssignVariableOp_14AssignVariableOp(assignvariableop_14_stream_conv2d_kernelIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15
AssignVariableOp_15AssignVariableOp&assignvariableop_15_stream_conv2d_biasIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16¥
AssignVariableOp_16AssignVariableOp,assignvariableop_16_stream_1_conv2d_1_kernelIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17£
AssignVariableOp_17AssignVariableOp*assignvariableop_17_stream_1_conv2d_1_biasIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18¥
AssignVariableOp_18AssignVariableOp,assignvariableop_18_stream_2_conv2d_2_kernelIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19£
AssignVariableOp_19AssignVariableOp*assignvariableop_19_stream_2_conv2d_2_biasIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20¥
AssignVariableOp_20AssignVariableOp,assignvariableop_20_stream_3_conv2d_3_kernelIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21£
AssignVariableOp_21AssignVariableOp*assignvariableop_21_stream_3_conv2d_3_biasIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22¥
AssignVariableOp_22AssignVariableOp,assignvariableop_22_stream_4_conv2d_4_kernelIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23£
AssignVariableOp_23AssignVariableOp*assignvariableop_23_stream_4_conv2d_4_biasIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24¥
AssignVariableOp_24AssignVariableOp,assignvariableop_24_stream_5_conv2d_5_kernelIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25£
AssignVariableOp_25AssignVariableOp*assignvariableop_25_stream_5_conv2d_5_biasIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26¥
AssignVariableOp_26AssignVariableOp,assignvariableop_26_stream_6_conv2d_6_kernelIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27£
AssignVariableOp_27AssignVariableOp*assignvariableop_27_stream_6_conv2d_6_biasIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27¨
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slicesÄ
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpÆ
Identity_28Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_28Ó
Identity_29IdentityIdentity_28:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_29"#
identity_29Identity_29:output:0*
_input_shapest
r: ::::::::::::::::::::::::::::2$
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
AssignVariableOp_27AssignVariableOp_272(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 


B__inference_stream_2_layer_call_and_return_conditional_losses_2914

inputs,
(readvariableop_streaming_stream_2_states;
7conv2d_2_conv2d_readvariableop_stream_2_conv2d_2_kernel:
6conv2d_2_biasadd_readvariableop_stream_2_conv2d_2_bias
identity¢AssignVariableOp
ReadVariableOpReadVariableOp(readvariableop_streaming_stream_2_states*&
_output_shapes
:@*
dtype02
ReadVariableOp
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:@*

begin_mask*
end_mask2
strided_slice\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2strided_slice:output:0inputsconcat/axis:output:0*
N*
T0*&
_output_shapes
:@2
concat§
AssignVariableOpAssignVariableOp(readvariableop_streaming_stream_2_statesconcat:output:0^ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOp
conv2d_2/dilation_rateConst^AssignVariableOp*
_output_shapes
:*
dtype0*
valueB"      2
conv2d_2/dilation_rateÓ
conv2d_2/Conv2D/ReadVariableOpReadVariableOp7conv2d_2_conv2d_readvariableop_stream_2_conv2d_2_kernel^AssignVariableOp*&
_output_shapes
:@@*
dtype02 
conv2d_2/Conv2D/ReadVariableOp¿
conv2d_2/Conv2DConv2Dconcat:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
conv2d_2/Conv2DÈ
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp6conv2d_2_biasadd_readvariableop_stream_2_conv2d_2_bias^AssignVariableOp*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp£
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
conv2d_2/BiasAddr
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*&
_output_shapes
:@2
conv2d_2/Relu
IdentityIdentityconv2d_2/Relu:activations:0^AssignVariableOp*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*1
_input_shapes 
:@:::2$
AssignVariableOpAssignVariableOp:N J
&
_output_shapes
:@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 


B__inference_stream_2_layer_call_and_return_conditional_losses_4217

inputs,
(readvariableop_streaming_stream_2_states;
7conv2d_2_conv2d_readvariableop_stream_2_conv2d_2_kernel:
6conv2d_2_biasadd_readvariableop_stream_2_conv2d_2_bias
identity¢AssignVariableOp
ReadVariableOpReadVariableOp(readvariableop_streaming_stream_2_states*&
_output_shapes
:@*
dtype02
ReadVariableOp
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:@*

begin_mask*
end_mask2
strided_slice\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2strided_slice:output:0inputsconcat/axis:output:0*
N*
T0*&
_output_shapes
:@2
concat§
AssignVariableOpAssignVariableOp(readvariableop_streaming_stream_2_statesconcat:output:0^ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOpÓ
conv2d_2/Conv2D/ReadVariableOpReadVariableOp7conv2d_2_conv2d_readvariableop_stream_2_conv2d_2_kernel^AssignVariableOp*&
_output_shapes
:@@*
dtype02 
conv2d_2/Conv2D/ReadVariableOp¿
conv2d_2/Conv2DConv2Dconcat:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
conv2d_2/Conv2DÈ
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp6conv2d_2_biasadd_readvariableop_stream_2_conv2d_2_bias^AssignVariableOp*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp£
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
conv2d_2/BiasAddr
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*&
_output_shapes
:@2
conv2d_2/Relu
IdentityIdentityconv2d_2/Relu:activations:0^AssignVariableOp*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*1
_input_shapes 
:@:::2$
AssignVariableOpAssignVariableOp:N J
&
_output_shapes
:@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ø

`
A__inference_dropout_layer_call_and_return_conditional_losses_3100

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constk
dropout/MulMulinputsdropout/Const:output:0*
T0*
_output_shapes
:	2
dropout/Mulo
dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     2
dropout/Shape¬
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
_output_shapes
:	*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y¶
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	2
dropout/GreaterEqualw
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	2
dropout/Castr
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*
_output_shapes
:	2
dropout/Mul_1]
IdentityIdentitydropout/Mul_1:z:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs

Á
?__inference_dense_layer_call_and_return_conditional_losses_4390

inputs0
,matmul_readvariableop_streaming_dense_kernel/
+biasadd_readvariableop_streaming_dense_bias
identity
MatMul/ReadVariableOpReadVariableOp,matmul_readvariableop_streaming_dense_kernel* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpk
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
MatMul
BiasAdd/ReadVariableOpReadVariableOp+biasadd_readvariableop_streaming_dense_bias*
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
:	:::G C

_output_shapes
:	
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
þ

+__inference_functional_1_layer_call_fn_4133
input_audio
streaming_stream_states
stream_conv2d_kernel
stream_conv2d_bias
streaming_stream_1_states
stream_1_conv2d_1_kernel
stream_1_conv2d_1_bias
streaming_stream_2_states
stream_2_conv2d_2_kernel
stream_2_conv2d_2_bias
streaming_stream_3_states
stream_3_conv2d_3_kernel
stream_3_conv2d_3_bias
streaming_stream_4_states
stream_4_conv2d_4_kernel
stream_4_conv2d_4_bias
streaming_stream_5_states
stream_5_conv2d_5_kernel
stream_5_conv2d_5_bias
streaming_stream_6_states
stream_6_conv2d_6_kernel
stream_6_conv2d_6_bias
streaming_stream_7_states
streaming_dense_kernel
streaming_dense_bias
streaming_dense_1_kernel
streaming_dense_1_bias
streaming_dense_2_kernel
streaming_dense_2_bias
identity¢StatefulPartitionedCallÅ
StatefulPartitionedCallStatefulPartitionedCallinput_audiostreaming_stream_statesstream_conv2d_kernelstream_conv2d_biasstreaming_stream_1_statesstream_1_conv2d_1_kernelstream_1_conv2d_1_biasstreaming_stream_2_statesstream_2_conv2d_2_kernelstream_2_conv2d_2_biasstreaming_stream_3_statesstream_3_conv2d_3_kernelstream_3_conv2d_3_biasstreaming_stream_4_statesstream_4_conv2d_4_kernelstream_4_conv2d_4_biasstreaming_stream_5_statesstream_5_conv2d_5_kernelstream_5_conv2d_5_biasstreaming_stream_6_statesstream_6_conv2d_6_kernelstream_6_conv2d_6_biasstreaming_stream_7_statesstreaming_dense_kernelstreaming_dense_biasstreaming_dense_1_kernelstreaming_dense_1_biasstreaming_dense_2_kernelstreaming_dense_2_bias*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*6
_read_only_resource_inputs
	**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_functional_1_layer_call_and_return_conditional_losses_33572
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinput_audio:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
¶

'__inference_conv2d_2_layer_call_fn_2676

inputs
stream_2_conv2d_2_kernel
stream_2_conv2d_2_bias
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsstream_2_conv2d_2_kernelstream_2_conv2d_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_26712
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 


B__inference_stream_1_layer_call_and_return_conditional_losses_4190

inputs,
(readvariableop_streaming_stream_1_states;
7conv2d_1_conv2d_readvariableop_stream_1_conv2d_1_kernel:
6conv2d_1_biasadd_readvariableop_stream_1_conv2d_1_bias
identity¢AssignVariableOp
ReadVariableOpReadVariableOp(readvariableop_streaming_stream_1_states*&
_output_shapes
:@*
dtype02
ReadVariableOp
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:@*

begin_mask*
end_mask2
strided_slice\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2strided_slice:output:0inputsconcat/axis:output:0*
N*
T0*&
_output_shapes
:@2
concat§
AssignVariableOpAssignVariableOp(readvariableop_streaming_stream_1_statesconcat:output:0^ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOpÓ
conv2d_1/Conv2D/ReadVariableOpReadVariableOp7conv2d_1_conv2d_readvariableop_stream_1_conv2d_1_kernel^AssignVariableOp*&
_output_shapes
:@@*
dtype02 
conv2d_1/Conv2D/ReadVariableOp¿
conv2d_1/Conv2DConv2Dconcat:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
conv2d_1/Conv2DÈ
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp6conv2d_1_biasadd_readvariableop_stream_1_conv2d_1_bias^AssignVariableOp*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp£
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
conv2d_1/BiasAddr
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*&
_output_shapes
:@2
conv2d_1/Relu
IdentityIdentityconv2d_1/Relu:activations:0^AssignVariableOp*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*1
_input_shapes 
:@:::2$
AssignVariableOpAssignVariableOp:N J
&
_output_shapes
:@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
©
¹
'__inference_stream_6_layer_call_fn_4333

inputs
streaming_stream_6_states
stream_6_conv2d_6_kernel
stream_6_conv2d_6_bias
identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsstreaming_stream_6_statesstream_6_conv2d_6_kernelstream_6_conv2d_6_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_stream_6_layer_call_and_return_conditional_losses_30502
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:2

Identity"
identityIdentity:output:0*1
_input_shapes 
:@:::22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ÄË

F__inference_functional_1_layer_call_and_return_conditional_losses_4067
input_audio1
-stream_readvariableop_streaming_stream_states<
8stream_conv2d_conv2d_readvariableop_stream_conv2d_kernel;
7stream_conv2d_biasadd_readvariableop_stream_conv2d_bias5
1stream_1_readvariableop_streaming_stream_1_statesD
@stream_1_conv2d_1_conv2d_readvariableop_stream_1_conv2d_1_kernelC
?stream_1_conv2d_1_biasadd_readvariableop_stream_1_conv2d_1_bias5
1stream_2_readvariableop_streaming_stream_2_statesD
@stream_2_conv2d_2_conv2d_readvariableop_stream_2_conv2d_2_kernelC
?stream_2_conv2d_2_biasadd_readvariableop_stream_2_conv2d_2_bias5
1stream_3_readvariableop_streaming_stream_3_statesD
@stream_3_conv2d_3_conv2d_readvariableop_stream_3_conv2d_3_kernelC
?stream_3_conv2d_3_biasadd_readvariableop_stream_3_conv2d_3_bias5
1stream_4_readvariableop_streaming_stream_4_statesD
@stream_4_conv2d_4_conv2d_readvariableop_stream_4_conv2d_4_kernelC
?stream_4_conv2d_4_biasadd_readvariableop_stream_4_conv2d_4_bias5
1stream_5_readvariableop_streaming_stream_5_statesD
@stream_5_conv2d_5_conv2d_readvariableop_stream_5_conv2d_5_kernelC
?stream_5_conv2d_5_biasadd_readvariableop_stream_5_conv2d_5_bias5
1stream_6_readvariableop_streaming_stream_6_statesD
@stream_6_conv2d_6_conv2d_readvariableop_stream_6_conv2d_6_kernelC
?stream_6_conv2d_6_biasadd_readvariableop_stream_6_conv2d_6_bias5
1stream_7_readvariableop_streaming_stream_7_states6
2dense_matmul_readvariableop_streaming_dense_kernel5
1dense_biasadd_readvariableop_streaming_dense_bias:
6dense_1_matmul_readvariableop_streaming_dense_1_kernel9
5dense_1_biasadd_readvariableop_streaming_dense_1_bias:
6dense_2_matmul_readvariableop_streaming_dense_2_kernel9
5dense_2_biasadd_readvariableop_streaming_dense_2_bias
identity¢stream/AssignVariableOp¢stream_1/AssignVariableOp¢stream_2/AssignVariableOp¢stream_3/AssignVariableOp¢stream_4/AssignVariableOp¢stream_5/AssignVariableOp¢stream_6/AssignVariableOp¢stream_7/AssignVariableOp
%tf_op_layer_ExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2'
%tf_op_layer_ExpandDims/ExpandDims/dimÑ
!tf_op_layer_ExpandDims/ExpandDims
ExpandDimsinput_audio.tf_op_layer_ExpandDims/ExpandDims/dim:output:0*
T0*
_cloned(*&
_output_shapes
:2#
!tf_op_layer_ExpandDims/ExpandDims¤
stream/ReadVariableOpReadVariableOp-stream_readvariableop_streaming_stream_states*&
_output_shapes
:*
dtype02
stream/ReadVariableOp
stream/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
stream/strided_slice/stack
stream/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
stream/strided_slice/stack_1
stream/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
stream/strided_slice/stack_2®
stream/strided_sliceStridedSlicestream/ReadVariableOp:value:0#stream/strided_slice/stack:output:0%stream/strided_slice/stack_1:output:0%stream/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:*

begin_mask*
end_mask2
stream/strided_slicej
stream/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
stream/concat/axisÌ
stream/concatConcatV2stream/strided_slice:output:0*tf_op_layer_ExpandDims/ExpandDims:output:0stream/concat/axis:output:0*
N*
T0*&
_output_shapes
:2
stream/concatÈ
stream/AssignVariableOpAssignVariableOp-stream_readvariableop_streaming_stream_statesstream/concat:output:0^stream/ReadVariableOp*
_output_shapes
 *
dtype02
stream/AssignVariableOpå
#stream/conv2d/Conv2D/ReadVariableOpReadVariableOp8stream_conv2d_conv2d_readvariableop_stream_conv2d_kernel^stream/AssignVariableOp*&
_output_shapes
:@*
dtype02%
#stream/conv2d/Conv2D/ReadVariableOpÕ
stream/conv2d/Conv2DConv2Dstream/concat:output:0+stream/conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
stream/conv2d/Conv2DÚ
$stream/conv2d/BiasAdd/ReadVariableOpReadVariableOp7stream_conv2d_biasadd_readvariableop_stream_conv2d_bias^stream/AssignVariableOp*
_output_shapes
:@*
dtype02&
$stream/conv2d/BiasAdd/ReadVariableOp·
stream/conv2d/BiasAddBiasAddstream/conv2d/Conv2D:output:0,stream/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
stream/conv2d/BiasAdd
stream/conv2d/ReluRelustream/conv2d/BiasAdd:output:0*
T0*&
_output_shapes
:@2
stream/conv2d/Relu¬
stream_1/ReadVariableOpReadVariableOp1stream_1_readvariableop_streaming_stream_1_states*&
_output_shapes
:@*
dtype02
stream_1/ReadVariableOp
stream_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
stream_1/strided_slice/stack
stream_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2 
stream_1/strided_slice/stack_1
stream_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2 
stream_1/strided_slice/stack_2º
stream_1/strided_sliceStridedSlicestream_1/ReadVariableOp:value:0%stream_1/strided_slice/stack:output:0'stream_1/strided_slice/stack_1:output:0'stream_1/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:@*

begin_mask*
end_mask2
stream_1/strided_slicen
stream_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
stream_1/concat/axisÊ
stream_1/concatConcatV2stream_1/strided_slice:output:0 stream/conv2d/Relu:activations:0stream_1/concat/axis:output:0*
N*
T0*&
_output_shapes
:@2
stream_1/concatÔ
stream_1/AssignVariableOpAssignVariableOp1stream_1_readvariableop_streaming_stream_1_statesstream_1/concat:output:0^stream_1/ReadVariableOp*
_output_shapes
 *
dtype02
stream_1/AssignVariableOp÷
'stream_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp@stream_1_conv2d_1_conv2d_readvariableop_stream_1_conv2d_1_kernel^stream_1/AssignVariableOp*&
_output_shapes
:@@*
dtype02)
'stream_1/conv2d_1/Conv2D/ReadVariableOpã
stream_1/conv2d_1/Conv2DConv2Dstream_1/concat:output:0/stream_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
stream_1/conv2d_1/Conv2Dì
(stream_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp?stream_1_conv2d_1_biasadd_readvariableop_stream_1_conv2d_1_bias^stream_1/AssignVariableOp*
_output_shapes
:@*
dtype02*
(stream_1/conv2d_1/BiasAdd/ReadVariableOpÇ
stream_1/conv2d_1/BiasAddBiasAdd!stream_1/conv2d_1/Conv2D:output:00stream_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
stream_1/conv2d_1/BiasAdd
stream_1/conv2d_1/ReluRelu"stream_1/conv2d_1/BiasAdd:output:0*
T0*&
_output_shapes
:@2
stream_1/conv2d_1/Relu¬
stream_2/ReadVariableOpReadVariableOp1stream_2_readvariableop_streaming_stream_2_states*&
_output_shapes
:@*
dtype02
stream_2/ReadVariableOp
stream_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
stream_2/strided_slice/stack
stream_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2 
stream_2/strided_slice/stack_1
stream_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2 
stream_2/strided_slice/stack_2º
stream_2/strided_sliceStridedSlicestream_2/ReadVariableOp:value:0%stream_2/strided_slice/stack:output:0'stream_2/strided_slice/stack_1:output:0'stream_2/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:@*

begin_mask*
end_mask2
stream_2/strided_slicen
stream_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
stream_2/concat/axisÎ
stream_2/concatConcatV2stream_2/strided_slice:output:0$stream_1/conv2d_1/Relu:activations:0stream_2/concat/axis:output:0*
N*
T0*&
_output_shapes
:@2
stream_2/concatÔ
stream_2/AssignVariableOpAssignVariableOp1stream_2_readvariableop_streaming_stream_2_statesstream_2/concat:output:0^stream_2/ReadVariableOp*
_output_shapes
 *
dtype02
stream_2/AssignVariableOp÷
'stream_2/conv2d_2/Conv2D/ReadVariableOpReadVariableOp@stream_2_conv2d_2_conv2d_readvariableop_stream_2_conv2d_2_kernel^stream_2/AssignVariableOp*&
_output_shapes
:@@*
dtype02)
'stream_2/conv2d_2/Conv2D/ReadVariableOpã
stream_2/conv2d_2/Conv2DConv2Dstream_2/concat:output:0/stream_2/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
stream_2/conv2d_2/Conv2Dì
(stream_2/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp?stream_2_conv2d_2_biasadd_readvariableop_stream_2_conv2d_2_bias^stream_2/AssignVariableOp*
_output_shapes
:@*
dtype02*
(stream_2/conv2d_2/BiasAdd/ReadVariableOpÇ
stream_2/conv2d_2/BiasAddBiasAdd!stream_2/conv2d_2/Conv2D:output:00stream_2/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
stream_2/conv2d_2/BiasAdd
stream_2/conv2d_2/ReluRelu"stream_2/conv2d_2/BiasAdd:output:0*
T0*&
_output_shapes
:@2
stream_2/conv2d_2/Relu¬
stream_3/ReadVariableOpReadVariableOp1stream_3_readvariableop_streaming_stream_3_states*&
_output_shapes
:@*
dtype02
stream_3/ReadVariableOp
stream_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
stream_3/strided_slice/stack
stream_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2 
stream_3/strided_slice/stack_1
stream_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2 
stream_3/strided_slice/stack_2º
stream_3/strided_sliceStridedSlicestream_3/ReadVariableOp:value:0%stream_3/strided_slice/stack:output:0'stream_3/strided_slice/stack_1:output:0'stream_3/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:@*

begin_mask*
end_mask2
stream_3/strided_slicen
stream_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
stream_3/concat/axisÎ
stream_3/concatConcatV2stream_3/strided_slice:output:0$stream_2/conv2d_2/Relu:activations:0stream_3/concat/axis:output:0*
N*
T0*&
_output_shapes
:@2
stream_3/concatÔ
stream_3/AssignVariableOpAssignVariableOp1stream_3_readvariableop_streaming_stream_3_statesstream_3/concat:output:0^stream_3/ReadVariableOp*
_output_shapes
 *
dtype02
stream_3/AssignVariableOp÷
'stream_3/conv2d_3/Conv2D/ReadVariableOpReadVariableOp@stream_3_conv2d_3_conv2d_readvariableop_stream_3_conv2d_3_kernel^stream_3/AssignVariableOp*&
_output_shapes
:@@*
dtype02)
'stream_3/conv2d_3/Conv2D/ReadVariableOpã
stream_3/conv2d_3/Conv2DConv2Dstream_3/concat:output:0/stream_3/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
stream_3/conv2d_3/Conv2Dì
(stream_3/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp?stream_3_conv2d_3_biasadd_readvariableop_stream_3_conv2d_3_bias^stream_3/AssignVariableOp*
_output_shapes
:@*
dtype02*
(stream_3/conv2d_3/BiasAdd/ReadVariableOpÇ
stream_3/conv2d_3/BiasAddBiasAdd!stream_3/conv2d_3/Conv2D:output:00stream_3/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
stream_3/conv2d_3/BiasAdd
stream_3/conv2d_3/ReluRelu"stream_3/conv2d_3/BiasAdd:output:0*
T0*&
_output_shapes
:@2
stream_3/conv2d_3/Relu¬
stream_4/ReadVariableOpReadVariableOp1stream_4_readvariableop_streaming_stream_4_states*&
_output_shapes
:@*
dtype02
stream_4/ReadVariableOp
stream_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
stream_4/strided_slice/stack
stream_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2 
stream_4/strided_slice/stack_1
stream_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2 
stream_4/strided_slice/stack_2º
stream_4/strided_sliceStridedSlicestream_4/ReadVariableOp:value:0%stream_4/strided_slice/stack:output:0'stream_4/strided_slice/stack_1:output:0'stream_4/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:@*

begin_mask*
end_mask2
stream_4/strided_slicen
stream_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
stream_4/concat/axisÎ
stream_4/concatConcatV2stream_4/strided_slice:output:0$stream_3/conv2d_3/Relu:activations:0stream_4/concat/axis:output:0*
N*
T0*&
_output_shapes
:@2
stream_4/concatÔ
stream_4/AssignVariableOpAssignVariableOp1stream_4_readvariableop_streaming_stream_4_statesstream_4/concat:output:0^stream_4/ReadVariableOp*
_output_shapes
 *
dtype02
stream_4/AssignVariableOpø
'stream_4/conv2d_4/Conv2D/ReadVariableOpReadVariableOp@stream_4_conv2d_4_conv2d_readvariableop_stream_4_conv2d_4_kernel^stream_4/AssignVariableOp*'
_output_shapes
:@*
dtype02)
'stream_4/conv2d_4/Conv2D/ReadVariableOpä
stream_4/conv2d_4/Conv2DConv2Dstream_4/concat:output:0/stream_4/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:*
paddingVALID*
strides
2
stream_4/conv2d_4/Conv2Dí
(stream_4/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp?stream_4_conv2d_4_biasadd_readvariableop_stream_4_conv2d_4_bias^stream_4/AssignVariableOp*
_output_shapes	
:*
dtype02*
(stream_4/conv2d_4/BiasAdd/ReadVariableOpÈ
stream_4/conv2d_4/BiasAddBiasAdd!stream_4/conv2d_4/Conv2D:output:00stream_4/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:2
stream_4/conv2d_4/BiasAdd
stream_4/conv2d_4/ReluRelu"stream_4/conv2d_4/BiasAdd:output:0*
T0*'
_output_shapes
:2
stream_4/conv2d_4/Relu­
stream_5/ReadVariableOpReadVariableOp1stream_5_readvariableop_streaming_stream_5_states*'
_output_shapes
:*
dtype02
stream_5/ReadVariableOp
stream_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
stream_5/strided_slice/stack
stream_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2 
stream_5/strided_slice/stack_1
stream_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2 
stream_5/strided_slice/stack_2»
stream_5/strided_sliceStridedSlicestream_5/ReadVariableOp:value:0%stream_5/strided_slice/stack:output:0'stream_5/strided_slice/stack_1:output:0'stream_5/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:*

begin_mask*
end_mask2
stream_5/strided_slicen
stream_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
stream_5/concat/axisÏ
stream_5/concatConcatV2stream_5/strided_slice:output:0$stream_4/conv2d_4/Relu:activations:0stream_5/concat/axis:output:0*
N*
T0*'
_output_shapes
:2
stream_5/concatÔ
stream_5/AssignVariableOpAssignVariableOp1stream_5_readvariableop_streaming_stream_5_statesstream_5/concat:output:0^stream_5/ReadVariableOp*
_output_shapes
 *
dtype02
stream_5/AssignVariableOpø
'stream_5/conv2d_5/Conv2D/ReadVariableOpReadVariableOp@stream_5_conv2d_5_conv2d_readvariableop_stream_5_conv2d_5_kernel^stream_5/AssignVariableOp*'
_output_shapes
:@*
dtype02)
'stream_5/conv2d_5/Conv2D/ReadVariableOpã
stream_5/conv2d_5/Conv2DConv2Dstream_5/concat:output:0/stream_5/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
stream_5/conv2d_5/Conv2Dì
(stream_5/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp?stream_5_conv2d_5_biasadd_readvariableop_stream_5_conv2d_5_bias^stream_5/AssignVariableOp*
_output_shapes
:@*
dtype02*
(stream_5/conv2d_5/BiasAdd/ReadVariableOpÇ
stream_5/conv2d_5/BiasAddBiasAdd!stream_5/conv2d_5/Conv2D:output:00stream_5/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
stream_5/conv2d_5/BiasAdd
stream_5/conv2d_5/ReluRelu"stream_5/conv2d_5/BiasAdd:output:0*
T0*&
_output_shapes
:@2
stream_5/conv2d_5/Relu¬
stream_6/ReadVariableOpReadVariableOp1stream_6_readvariableop_streaming_stream_6_states*&
_output_shapes
:@*
dtype02
stream_6/ReadVariableOp
stream_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
stream_6/strided_slice/stack
stream_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2 
stream_6/strided_slice/stack_1
stream_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2 
stream_6/strided_slice/stack_2º
stream_6/strided_sliceStridedSlicestream_6/ReadVariableOp:value:0%stream_6/strided_slice/stack:output:0'stream_6/strided_slice/stack_1:output:0'stream_6/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:@*

begin_mask*
end_mask2
stream_6/strided_slicen
stream_6/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
stream_6/concat/axisÎ
stream_6/concatConcatV2stream_6/strided_slice:output:0$stream_5/conv2d_5/Relu:activations:0stream_6/concat/axis:output:0*
N*
T0*&
_output_shapes
:@2
stream_6/concatÔ
stream_6/AssignVariableOpAssignVariableOp1stream_6_readvariableop_streaming_stream_6_statesstream_6/concat:output:0^stream_6/ReadVariableOp*
_output_shapes
 *
dtype02
stream_6/AssignVariableOpø
'stream_6/conv2d_6/Conv2D/ReadVariableOpReadVariableOp@stream_6_conv2d_6_conv2d_readvariableop_stream_6_conv2d_6_kernel^stream_6/AssignVariableOp*'
_output_shapes
:@*
dtype02)
'stream_6/conv2d_6/Conv2D/ReadVariableOpä
stream_6/conv2d_6/Conv2DConv2Dstream_6/concat:output:0/stream_6/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:*
paddingVALID*
strides
2
stream_6/conv2d_6/Conv2Dí
(stream_6/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp?stream_6_conv2d_6_biasadd_readvariableop_stream_6_conv2d_6_bias^stream_6/AssignVariableOp*
_output_shapes	
:*
dtype02*
(stream_6/conv2d_6/BiasAdd/ReadVariableOpÈ
stream_6/conv2d_6/BiasAddBiasAdd!stream_6/conv2d_6/Conv2D:output:00stream_6/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:2
stream_6/conv2d_6/BiasAdd
stream_6/conv2d_6/ReluRelu"stream_6/conv2d_6/BiasAdd:output:0*
T0*'
_output_shapes
:2
stream_6/conv2d_6/Relu­
stream_7/ReadVariableOpReadVariableOp1stream_7_readvariableop_streaming_stream_7_states*'
_output_shapes
:*
dtype02
stream_7/ReadVariableOp
stream_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
stream_7/strided_slice/stack
stream_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2 
stream_7/strided_slice/stack_1
stream_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2 
stream_7/strided_slice/stack_2¹
stream_7/strided_sliceStridedSlicestream_7/ReadVariableOp:value:0%stream_7/strided_slice/stack:output:0'stream_7/strided_slice/stack_1:output:0'stream_7/strided_slice/stack_2:output:0*
Index0*
T0*%
_output_shapes
: *

begin_mask*
end_mask2
stream_7/strided_slicen
stream_7/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
stream_7/concat/axisÏ
stream_7/concatConcatV2stream_7/strided_slice:output:0$stream_6/conv2d_6/Relu:activations:0stream_7/concat/axis:output:0*
N*
T0*'
_output_shapes
:2
stream_7/concatÔ
stream_7/AssignVariableOpAssignVariableOp1stream_7_readvariableop_streaming_stream_7_statesstream_7/concat:output:0^stream_7/ReadVariableOp*
_output_shapes
 *
dtype02
stream_7/AssignVariableOp
stream_7/flatten/ConstConst^stream_7/AssignVariableOp*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
stream_7/flatten/Const¤
stream_7/flatten/ReshapeReshapestream_7/concat:output:0stream_7/flatten/Const:output:0*
T0*
_output_shapes
:	2
stream_7/flatten/Reshape}
dropout/IdentityIdentity!stream_7/flatten/Reshape:output:0*
T0*
_output_shapes
:	2
dropout/Identity¯
dense/MatMul/ReadVariableOpReadVariableOp2dense_matmul_readvariableop_streaming_dense_kernel* 
_output_shapes
:
*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMuldropout/Identity:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense/MatMul«
dense/BiasAdd/ReadVariableOpReadVariableOp1dense_biasadd_readvariableop_streaming_dense_bias*
_output_shapes	
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense/BiasAdd·
dense_1/MatMul/ReadVariableOpReadVariableOp6dense_1_matmul_readvariableop_streaming_dense_1_kernel* 
_output_shapes
:
*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldense/BiasAdd:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense_1/MatMul³
dense_1/BiasAdd/ReadVariableOpReadVariableOp5dense_1_biasadd_readvariableop_streaming_dense_1_bias*
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
dense_1/Relu¶
dense_2/MatMul/ReadVariableOpReadVariableOp6dense_2_matmul_readvariableop_streaming_dense_2_kernel*
_output_shapes
:	*
dtype02
dense_2/MatMul/ReadVariableOp
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_2/MatMul²
dense_2/BiasAdd/ReadVariableOpReadVariableOp5dense_2_biasadd_readvariableop_streaming_dense_2_bias*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_2/BiasAddÁ
IdentityIdentitydense_2/BiasAdd:output:0^stream/AssignVariableOp^stream_1/AssignVariableOp^stream_2/AssignVariableOp^stream_3/AssignVariableOp^stream_4/AssignVariableOp^stream_5/AssignVariableOp^stream_6/AssignVariableOp^stream_7/AssignVariableOp*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*
_input_shapes
~:::::::::::::::::::::::::::::22
stream/AssignVariableOpstream/AssignVariableOp26
stream_1/AssignVariableOpstream_1/AssignVariableOp26
stream_2/AssignVariableOpstream_2/AssignVariableOp26
stream_3/AssignVariableOpstream_3/AssignVariableOp26
stream_4/AssignVariableOpstream_4/AssignVariableOp26
stream_5/AssignVariableOpstream_5/AssignVariableOp26
stream_6/AssignVariableOpstream_6/AssignVariableOp26
stream_7/AssignVariableOpstream_7/AssignVariableOp:X T
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinput_audio:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Þ
È
B__inference_conv2d_3_layer_call_and_return_conditional_losses_2689

inputs2
.conv2d_readvariableop_stream_3_conv2d_3_kernel1
-biasadd_readvariableop_stream_3_conv2d_3_bias
identityo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate¥
Conv2D/ReadVariableOpReadVariableOp.conv2d_readvariableop_stream_3_conv2d_3_kernel*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¶
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOp-biasadd_readvariableop_stream_3_conv2d_3_bias*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2
Relu
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Ô

@__inference_stream_layer_call_and_return_conditional_losses_2846

inputs*
&readvariableop_streaming_stream_states5
1conv2d_conv2d_readvariableop_stream_conv2d_kernel4
0conv2d_biasadd_readvariableop_stream_conv2d_bias
identity¢AssignVariableOp
ReadVariableOpReadVariableOp&readvariableop_streaming_stream_states*&
_output_shapes
:*
dtype02
ReadVariableOp
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:*

begin_mask*
end_mask2
strided_slice\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2strided_slice:output:0inputsconcat/axis:output:0*
N*
T0*&
_output_shapes
:2
concat¥
AssignVariableOpAssignVariableOp&readvariableop_streaming_stream_statesconcat:output:0^ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOp
conv2d/dilation_rateConst^AssignVariableOp*
_output_shapes
:*
dtype0*
valueB"      2
conv2d/dilation_rateÉ
conv2d/Conv2D/ReadVariableOpReadVariableOp1conv2d_conv2d_readvariableop_stream_conv2d_kernel^AssignVariableOp*&
_output_shapes
:@*
dtype02
conv2d/Conv2D/ReadVariableOp¹
conv2d/Conv2DConv2Dconcat:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
conv2d/Conv2D¾
conv2d/BiasAdd/ReadVariableOpReadVariableOp0conv2d_biasadd_readvariableop_stream_conv2d_bias^AssignVariableOp*
_output_shapes
:@*
dtype02
conv2d/BiasAdd/ReadVariableOp
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
conv2d/BiasAddl
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*&
_output_shapes
:@2
conv2d/Relu
IdentityIdentityconv2d/Relu:activations:0^AssignVariableOp*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*1
_input_shapes 
::::2$
AssignVariableOpAssignVariableOp:N J
&
_output_shapes
:
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 


B__inference_stream_5_layer_call_and_return_conditional_losses_4298

inputs,
(readvariableop_streaming_stream_5_states;
7conv2d_5_conv2d_readvariableop_stream_5_conv2d_5_kernel:
6conv2d_5_biasadd_readvariableop_stream_5_conv2d_5_bias
identity¢AssignVariableOp
ReadVariableOpReadVariableOp(readvariableop_streaming_stream_5_states*'
_output_shapes
:*
dtype02
ReadVariableOp
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:*

begin_mask*
end_mask2
strided_slice\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2strided_slice:output:0inputsconcat/axis:output:0*
N*
T0*'
_output_shapes
:2
concat§
AssignVariableOpAssignVariableOp(readvariableop_streaming_stream_5_statesconcat:output:0^ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOpÔ
conv2d_5/Conv2D/ReadVariableOpReadVariableOp7conv2d_5_conv2d_readvariableop_stream_5_conv2d_5_kernel^AssignVariableOp*'
_output_shapes
:@*
dtype02 
conv2d_5/Conv2D/ReadVariableOp¿
conv2d_5/Conv2DConv2Dconcat:output:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
conv2d_5/Conv2DÈ
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp6conv2d_5_biasadd_readvariableop_stream_5_conv2d_5_bias^AssignVariableOp*
_output_shapes
:@*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp£
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
conv2d_5/BiasAddr
conv2d_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*&
_output_shapes
:@2
conv2d_5/Relu
IdentityIdentityconv2d_5/Relu:activations:0^AssignVariableOp*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*2
_input_shapes!
::::2$
AssignVariableOpAssignVariableOp:O K
'
_output_shapes
:
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
§
¹
'__inference_stream_3_layer_call_fn_4252

inputs
streaming_stream_3_states
stream_3_conv2d_3_kernel
stream_3_conv2d_3_bias
identity¢StatefulPartitionedCall¨
StatefulPartitionedCallStatefulPartitionedCallinputsstreaming_stream_3_statesstream_3_conv2d_3_kernelstream_3_conv2d_3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_stream_3_layer_call_and_return_conditional_losses_29482
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*1
_input_shapes 
:@:::22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
»

"__inference_signature_wrapper_3423
input_audio
streaming_stream_states
stream_conv2d_kernel
stream_conv2d_bias
streaming_stream_1_states
stream_1_conv2d_1_kernel
stream_1_conv2d_1_bias
streaming_stream_2_states
stream_2_conv2d_2_kernel
stream_2_conv2d_2_bias
streaming_stream_3_states
stream_3_conv2d_3_kernel
stream_3_conv2d_3_bias
streaming_stream_4_states
stream_4_conv2d_4_kernel
stream_4_conv2d_4_bias
streaming_stream_5_states
stream_5_conv2d_5_kernel
stream_5_conv2d_5_bias
streaming_stream_6_states
stream_6_conv2d_6_kernel
stream_6_conv2d_6_bias
streaming_stream_7_states
streaming_dense_kernel
streaming_dense_bias
streaming_dense_1_kernel
streaming_dense_1_bias
streaming_dense_2_kernel
streaming_dense_2_bias
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_audiostreaming_stream_statesstream_conv2d_kernelstream_conv2d_biasstreaming_stream_1_statesstream_1_conv2d_1_kernelstream_1_conv2d_1_biasstreaming_stream_2_statesstream_2_conv2d_2_kernelstream_2_conv2d_2_biasstreaming_stream_3_statesstream_3_conv2d_3_kernelstream_3_conv2d_3_biasstreaming_stream_4_statesstream_4_conv2d_4_kernelstream_4_conv2d_4_biasstreaming_stream_5_statesstream_5_conv2d_5_kernelstream_5_conv2d_5_biasstreaming_stream_6_statesstream_6_conv2d_6_kernelstream_6_conv2d_6_biasstreaming_stream_7_statesstreaming_dense_kernelstreaming_dense_biasstreaming_dense_1_kernelstreaming_dense_1_biasstreaming_dense_2_kernelstreaming_dense_2_bias*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*6
_read_only_resource_inputs
	**
config_proto

CPU

GPU 2J 8*(
f#R!
__inference__wrapped_model_25772
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*
_input_shapes
~:::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
"
_output_shapes
:
%
_user_specified_nameinput_audio:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
 

$__inference_dense_layer_call_fn_4397

inputs
streaming_dense_kernel
streaming_dense_bias
identity¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCallinputsstreaming_dense_kernelstreaming_dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_31282
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*&
_input_shapes
:	::22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 

Á
?__inference_dense_layer_call_and_return_conditional_losses_3128

inputs0
,matmul_readvariableop_streaming_dense_kernel/
+biasadd_readvariableop_streaming_dense_bias
identity
MatMul/ReadVariableOpReadVariableOp,matmul_readvariableop_streaming_dense_kernel* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpk
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
MatMul
BiasAdd/ReadVariableOpReadVariableOp+biasadd_readvariableop_streaming_dense_bias*
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
:	:::G C

_output_shapes
:	
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 

Ç
A__inference_dense_2_layer_call_and_return_conditional_losses_3173

inputs2
.matmul_readvariableop_streaming_dense_2_kernel1
-biasadd_readvariableop_streaming_dense_2_bias
identity
MatMul/ReadVariableOpReadVariableOp.matmul_readvariableop_streaming_dense_2_kernel*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
MatMul
BiasAdd/ReadVariableOpReadVariableOp-biasadd_readvariableop_streaming_dense_2_bias*
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
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 


B__inference_stream_6_layer_call_and_return_conditional_losses_3050

inputs,
(readvariableop_streaming_stream_6_states;
7conv2d_6_conv2d_readvariableop_stream_6_conv2d_6_kernel:
6conv2d_6_biasadd_readvariableop_stream_6_conv2d_6_bias
identity¢AssignVariableOp
ReadVariableOpReadVariableOp(readvariableop_streaming_stream_6_states*&
_output_shapes
:@*
dtype02
ReadVariableOp
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:@*

begin_mask*
end_mask2
strided_slice\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2strided_slice:output:0inputsconcat/axis:output:0*
N*
T0*&
_output_shapes
:@2
concat§
AssignVariableOpAssignVariableOp(readvariableop_streaming_stream_6_statesconcat:output:0^ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOp
conv2d_6/dilation_rateConst^AssignVariableOp*
_output_shapes
:*
dtype0*
valueB"      2
conv2d_6/dilation_rateÔ
conv2d_6/Conv2D/ReadVariableOpReadVariableOp7conv2d_6_conv2d_readvariableop_stream_6_conv2d_6_kernel^AssignVariableOp*'
_output_shapes
:@*
dtype02 
conv2d_6/Conv2D/ReadVariableOpÀ
conv2d_6/Conv2DConv2Dconcat:output:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:*
paddingVALID*
strides
2
conv2d_6/Conv2DÉ
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp6conv2d_6_biasadd_readvariableop_stream_6_conv2d_6_bias^AssignVariableOp*
_output_shapes	
:*
dtype02!
conv2d_6/BiasAdd/ReadVariableOp¤
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:2
conv2d_6/BiasAdds
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*'
_output_shapes
:2
conv2d_6/Relu
IdentityIdentityconv2d_6/Relu:activations:0^AssignVariableOp*
T0*'
_output_shapes
:2

Identity"
identityIdentity:output:0*1
_input_shapes 
:@:::2$
AssignVariableOpAssignVariableOp:N J
&
_output_shapes
:@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
û
Ã
__inference__wrapped_model_2577
input_audio>
:functional_1_stream_readvariableop_streaming_stream_statesI
Efunctional_1_stream_conv2d_conv2d_readvariableop_stream_conv2d_kernelH
Dfunctional_1_stream_conv2d_biasadd_readvariableop_stream_conv2d_biasB
>functional_1_stream_1_readvariableop_streaming_stream_1_statesQ
Mfunctional_1_stream_1_conv2d_1_conv2d_readvariableop_stream_1_conv2d_1_kernelP
Lfunctional_1_stream_1_conv2d_1_biasadd_readvariableop_stream_1_conv2d_1_biasB
>functional_1_stream_2_readvariableop_streaming_stream_2_statesQ
Mfunctional_1_stream_2_conv2d_2_conv2d_readvariableop_stream_2_conv2d_2_kernelP
Lfunctional_1_stream_2_conv2d_2_biasadd_readvariableop_stream_2_conv2d_2_biasB
>functional_1_stream_3_readvariableop_streaming_stream_3_statesQ
Mfunctional_1_stream_3_conv2d_3_conv2d_readvariableop_stream_3_conv2d_3_kernelP
Lfunctional_1_stream_3_conv2d_3_biasadd_readvariableop_stream_3_conv2d_3_biasB
>functional_1_stream_4_readvariableop_streaming_stream_4_statesQ
Mfunctional_1_stream_4_conv2d_4_conv2d_readvariableop_stream_4_conv2d_4_kernelP
Lfunctional_1_stream_4_conv2d_4_biasadd_readvariableop_stream_4_conv2d_4_biasB
>functional_1_stream_5_readvariableop_streaming_stream_5_statesQ
Mfunctional_1_stream_5_conv2d_5_conv2d_readvariableop_stream_5_conv2d_5_kernelP
Lfunctional_1_stream_5_conv2d_5_biasadd_readvariableop_stream_5_conv2d_5_biasB
>functional_1_stream_6_readvariableop_streaming_stream_6_statesQ
Mfunctional_1_stream_6_conv2d_6_conv2d_readvariableop_stream_6_conv2d_6_kernelP
Lfunctional_1_stream_6_conv2d_6_biasadd_readvariableop_stream_6_conv2d_6_biasB
>functional_1_stream_7_readvariableop_streaming_stream_7_statesC
?functional_1_dense_matmul_readvariableop_streaming_dense_kernelB
>functional_1_dense_biasadd_readvariableop_streaming_dense_biasG
Cfunctional_1_dense_1_matmul_readvariableop_streaming_dense_1_kernelF
Bfunctional_1_dense_1_biasadd_readvariableop_streaming_dense_1_biasG
Cfunctional_1_dense_2_matmul_readvariableop_streaming_dense_2_kernelF
Bfunctional_1_dense_2_biasadd_readvariableop_streaming_dense_2_bias
identity¢$functional_1/stream/AssignVariableOp¢&functional_1/stream_1/AssignVariableOp¢&functional_1/stream_2/AssignVariableOp¢&functional_1/stream_3/AssignVariableOp¢&functional_1/stream_4/AssignVariableOp¢&functional_1/stream_5/AssignVariableOp¢&functional_1/stream_6/AssignVariableOp¢&functional_1/stream_7/AssignVariableOp³
2functional_1/tf_op_layer_ExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ24
2functional_1/tf_op_layer_ExpandDims/ExpandDims/dimø
.functional_1/tf_op_layer_ExpandDims/ExpandDims
ExpandDimsinput_audio;functional_1/tf_op_layer_ExpandDims/ExpandDims/dim:output:0*
T0*
_cloned(*&
_output_shapes
:20
.functional_1/tf_op_layer_ExpandDims/ExpandDimsË
"functional_1/stream/ReadVariableOpReadVariableOp:functional_1_stream_readvariableop_streaming_stream_states*&
_output_shapes
:*
dtype02$
"functional_1/stream/ReadVariableOp§
'functional_1/stream/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2)
'functional_1/stream/strided_slice/stack«
)functional_1/stream/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2+
)functional_1/stream/strided_slice/stack_1«
)functional_1/stream/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2+
)functional_1/stream/strided_slice/stack_2ü
!functional_1/stream/strided_sliceStridedSlice*functional_1/stream/ReadVariableOp:value:00functional_1/stream/strided_slice/stack:output:02functional_1/stream/strided_slice/stack_1:output:02functional_1/stream/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:*

begin_mask*
end_mask2#
!functional_1/stream/strided_slice
functional_1/stream/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2!
functional_1/stream/concat/axis
functional_1/stream/concatConcatV2*functional_1/stream/strided_slice:output:07functional_1/tf_op_layer_ExpandDims/ExpandDims:output:0(functional_1/stream/concat/axis:output:0*
N*
T0*&
_output_shapes
:2
functional_1/stream/concat
$functional_1/stream/AssignVariableOpAssignVariableOp:functional_1_stream_readvariableop_streaming_stream_states#functional_1/stream/concat:output:0#^functional_1/stream/ReadVariableOp*
_output_shapes
 *
dtype02&
$functional_1/stream/AssignVariableOp
0functional_1/stream/conv2d/Conv2D/ReadVariableOpReadVariableOpEfunctional_1_stream_conv2d_conv2d_readvariableop_stream_conv2d_kernel%^functional_1/stream/AssignVariableOp*&
_output_shapes
:@*
dtype022
0functional_1/stream/conv2d/Conv2D/ReadVariableOp
!functional_1/stream/conv2d/Conv2DConv2D#functional_1/stream/concat:output:08functional_1/stream/conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2#
!functional_1/stream/conv2d/Conv2D
1functional_1/stream/conv2d/BiasAdd/ReadVariableOpReadVariableOpDfunctional_1_stream_conv2d_biasadd_readvariableop_stream_conv2d_bias%^functional_1/stream/AssignVariableOp*
_output_shapes
:@*
dtype023
1functional_1/stream/conv2d/BiasAdd/ReadVariableOpë
"functional_1/stream/conv2d/BiasAddBiasAdd*functional_1/stream/conv2d/Conv2D:output:09functional_1/stream/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2$
"functional_1/stream/conv2d/BiasAdd¨
functional_1/stream/conv2d/ReluRelu+functional_1/stream/conv2d/BiasAdd:output:0*
T0*&
_output_shapes
:@2!
functional_1/stream/conv2d/ReluÓ
$functional_1/stream_1/ReadVariableOpReadVariableOp>functional_1_stream_1_readvariableop_streaming_stream_1_states*&
_output_shapes
:@*
dtype02&
$functional_1/stream_1/ReadVariableOp«
)functional_1/stream_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2+
)functional_1/stream_1/strided_slice/stack¯
+functional_1/stream_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2-
+functional_1/stream_1/strided_slice/stack_1¯
+functional_1/stream_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2-
+functional_1/stream_1/strided_slice/stack_2
#functional_1/stream_1/strided_sliceStridedSlice,functional_1/stream_1/ReadVariableOp:value:02functional_1/stream_1/strided_slice/stack:output:04functional_1/stream_1/strided_slice/stack_1:output:04functional_1/stream_1/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:@*

begin_mask*
end_mask2%
#functional_1/stream_1/strided_slice
!functional_1/stream_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!functional_1/stream_1/concat/axis
functional_1/stream_1/concatConcatV2,functional_1/stream_1/strided_slice:output:0-functional_1/stream/conv2d/Relu:activations:0*functional_1/stream_1/concat/axis:output:0*
N*
T0*&
_output_shapes
:@2
functional_1/stream_1/concat
&functional_1/stream_1/AssignVariableOpAssignVariableOp>functional_1_stream_1_readvariableop_streaming_stream_1_states%functional_1/stream_1/concat:output:0%^functional_1/stream_1/ReadVariableOp*
_output_shapes
 *
dtype02(
&functional_1/stream_1/AssignVariableOp«
4functional_1/stream_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOpMfunctional_1_stream_1_conv2d_1_conv2d_readvariableop_stream_1_conv2d_1_kernel'^functional_1/stream_1/AssignVariableOp*&
_output_shapes
:@@*
dtype026
4functional_1/stream_1/conv2d_1/Conv2D/ReadVariableOp
%functional_1/stream_1/conv2d_1/Conv2DConv2D%functional_1/stream_1/concat:output:0<functional_1/stream_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2'
%functional_1/stream_1/conv2d_1/Conv2D 
5functional_1/stream_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpLfunctional_1_stream_1_conv2d_1_biasadd_readvariableop_stream_1_conv2d_1_bias'^functional_1/stream_1/AssignVariableOp*
_output_shapes
:@*
dtype027
5functional_1/stream_1/conv2d_1/BiasAdd/ReadVariableOpû
&functional_1/stream_1/conv2d_1/BiasAddBiasAdd.functional_1/stream_1/conv2d_1/Conv2D:output:0=functional_1/stream_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2(
&functional_1/stream_1/conv2d_1/BiasAdd´
#functional_1/stream_1/conv2d_1/ReluRelu/functional_1/stream_1/conv2d_1/BiasAdd:output:0*
T0*&
_output_shapes
:@2%
#functional_1/stream_1/conv2d_1/ReluÓ
$functional_1/stream_2/ReadVariableOpReadVariableOp>functional_1_stream_2_readvariableop_streaming_stream_2_states*&
_output_shapes
:@*
dtype02&
$functional_1/stream_2/ReadVariableOp«
)functional_1/stream_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2+
)functional_1/stream_2/strided_slice/stack¯
+functional_1/stream_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2-
+functional_1/stream_2/strided_slice/stack_1¯
+functional_1/stream_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2-
+functional_1/stream_2/strided_slice/stack_2
#functional_1/stream_2/strided_sliceStridedSlice,functional_1/stream_2/ReadVariableOp:value:02functional_1/stream_2/strided_slice/stack:output:04functional_1/stream_2/strided_slice/stack_1:output:04functional_1/stream_2/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:@*

begin_mask*
end_mask2%
#functional_1/stream_2/strided_slice
!functional_1/stream_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!functional_1/stream_2/concat/axis
functional_1/stream_2/concatConcatV2,functional_1/stream_2/strided_slice:output:01functional_1/stream_1/conv2d_1/Relu:activations:0*functional_1/stream_2/concat/axis:output:0*
N*
T0*&
_output_shapes
:@2
functional_1/stream_2/concat
&functional_1/stream_2/AssignVariableOpAssignVariableOp>functional_1_stream_2_readvariableop_streaming_stream_2_states%functional_1/stream_2/concat:output:0%^functional_1/stream_2/ReadVariableOp*
_output_shapes
 *
dtype02(
&functional_1/stream_2/AssignVariableOp«
4functional_1/stream_2/conv2d_2/Conv2D/ReadVariableOpReadVariableOpMfunctional_1_stream_2_conv2d_2_conv2d_readvariableop_stream_2_conv2d_2_kernel'^functional_1/stream_2/AssignVariableOp*&
_output_shapes
:@@*
dtype026
4functional_1/stream_2/conv2d_2/Conv2D/ReadVariableOp
%functional_1/stream_2/conv2d_2/Conv2DConv2D%functional_1/stream_2/concat:output:0<functional_1/stream_2/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2'
%functional_1/stream_2/conv2d_2/Conv2D 
5functional_1/stream_2/conv2d_2/BiasAdd/ReadVariableOpReadVariableOpLfunctional_1_stream_2_conv2d_2_biasadd_readvariableop_stream_2_conv2d_2_bias'^functional_1/stream_2/AssignVariableOp*
_output_shapes
:@*
dtype027
5functional_1/stream_2/conv2d_2/BiasAdd/ReadVariableOpû
&functional_1/stream_2/conv2d_2/BiasAddBiasAdd.functional_1/stream_2/conv2d_2/Conv2D:output:0=functional_1/stream_2/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2(
&functional_1/stream_2/conv2d_2/BiasAdd´
#functional_1/stream_2/conv2d_2/ReluRelu/functional_1/stream_2/conv2d_2/BiasAdd:output:0*
T0*&
_output_shapes
:@2%
#functional_1/stream_2/conv2d_2/ReluÓ
$functional_1/stream_3/ReadVariableOpReadVariableOp>functional_1_stream_3_readvariableop_streaming_stream_3_states*&
_output_shapes
:@*
dtype02&
$functional_1/stream_3/ReadVariableOp«
)functional_1/stream_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2+
)functional_1/stream_3/strided_slice/stack¯
+functional_1/stream_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2-
+functional_1/stream_3/strided_slice/stack_1¯
+functional_1/stream_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2-
+functional_1/stream_3/strided_slice/stack_2
#functional_1/stream_3/strided_sliceStridedSlice,functional_1/stream_3/ReadVariableOp:value:02functional_1/stream_3/strided_slice/stack:output:04functional_1/stream_3/strided_slice/stack_1:output:04functional_1/stream_3/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:@*

begin_mask*
end_mask2%
#functional_1/stream_3/strided_slice
!functional_1/stream_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!functional_1/stream_3/concat/axis
functional_1/stream_3/concatConcatV2,functional_1/stream_3/strided_slice:output:01functional_1/stream_2/conv2d_2/Relu:activations:0*functional_1/stream_3/concat/axis:output:0*
N*
T0*&
_output_shapes
:@2
functional_1/stream_3/concat
&functional_1/stream_3/AssignVariableOpAssignVariableOp>functional_1_stream_3_readvariableop_streaming_stream_3_states%functional_1/stream_3/concat:output:0%^functional_1/stream_3/ReadVariableOp*
_output_shapes
 *
dtype02(
&functional_1/stream_3/AssignVariableOp«
4functional_1/stream_3/conv2d_3/Conv2D/ReadVariableOpReadVariableOpMfunctional_1_stream_3_conv2d_3_conv2d_readvariableop_stream_3_conv2d_3_kernel'^functional_1/stream_3/AssignVariableOp*&
_output_shapes
:@@*
dtype026
4functional_1/stream_3/conv2d_3/Conv2D/ReadVariableOp
%functional_1/stream_3/conv2d_3/Conv2DConv2D%functional_1/stream_3/concat:output:0<functional_1/stream_3/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2'
%functional_1/stream_3/conv2d_3/Conv2D 
5functional_1/stream_3/conv2d_3/BiasAdd/ReadVariableOpReadVariableOpLfunctional_1_stream_3_conv2d_3_biasadd_readvariableop_stream_3_conv2d_3_bias'^functional_1/stream_3/AssignVariableOp*
_output_shapes
:@*
dtype027
5functional_1/stream_3/conv2d_3/BiasAdd/ReadVariableOpû
&functional_1/stream_3/conv2d_3/BiasAddBiasAdd.functional_1/stream_3/conv2d_3/Conv2D:output:0=functional_1/stream_3/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2(
&functional_1/stream_3/conv2d_3/BiasAdd´
#functional_1/stream_3/conv2d_3/ReluRelu/functional_1/stream_3/conv2d_3/BiasAdd:output:0*
T0*&
_output_shapes
:@2%
#functional_1/stream_3/conv2d_3/ReluÓ
$functional_1/stream_4/ReadVariableOpReadVariableOp>functional_1_stream_4_readvariableop_streaming_stream_4_states*&
_output_shapes
:@*
dtype02&
$functional_1/stream_4/ReadVariableOp«
)functional_1/stream_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2+
)functional_1/stream_4/strided_slice/stack¯
+functional_1/stream_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2-
+functional_1/stream_4/strided_slice/stack_1¯
+functional_1/stream_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2-
+functional_1/stream_4/strided_slice/stack_2
#functional_1/stream_4/strided_sliceStridedSlice,functional_1/stream_4/ReadVariableOp:value:02functional_1/stream_4/strided_slice/stack:output:04functional_1/stream_4/strided_slice/stack_1:output:04functional_1/stream_4/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:@*

begin_mask*
end_mask2%
#functional_1/stream_4/strided_slice
!functional_1/stream_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!functional_1/stream_4/concat/axis
functional_1/stream_4/concatConcatV2,functional_1/stream_4/strided_slice:output:01functional_1/stream_3/conv2d_3/Relu:activations:0*functional_1/stream_4/concat/axis:output:0*
N*
T0*&
_output_shapes
:@2
functional_1/stream_4/concat
&functional_1/stream_4/AssignVariableOpAssignVariableOp>functional_1_stream_4_readvariableop_streaming_stream_4_states%functional_1/stream_4/concat:output:0%^functional_1/stream_4/ReadVariableOp*
_output_shapes
 *
dtype02(
&functional_1/stream_4/AssignVariableOp¬
4functional_1/stream_4/conv2d_4/Conv2D/ReadVariableOpReadVariableOpMfunctional_1_stream_4_conv2d_4_conv2d_readvariableop_stream_4_conv2d_4_kernel'^functional_1/stream_4/AssignVariableOp*'
_output_shapes
:@*
dtype026
4functional_1/stream_4/conv2d_4/Conv2D/ReadVariableOp
%functional_1/stream_4/conv2d_4/Conv2DConv2D%functional_1/stream_4/concat:output:0<functional_1/stream_4/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:*
paddingVALID*
strides
2'
%functional_1/stream_4/conv2d_4/Conv2D¡
5functional_1/stream_4/conv2d_4/BiasAdd/ReadVariableOpReadVariableOpLfunctional_1_stream_4_conv2d_4_biasadd_readvariableop_stream_4_conv2d_4_bias'^functional_1/stream_4/AssignVariableOp*
_output_shapes	
:*
dtype027
5functional_1/stream_4/conv2d_4/BiasAdd/ReadVariableOpü
&functional_1/stream_4/conv2d_4/BiasAddBiasAdd.functional_1/stream_4/conv2d_4/Conv2D:output:0=functional_1/stream_4/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:2(
&functional_1/stream_4/conv2d_4/BiasAddµ
#functional_1/stream_4/conv2d_4/ReluRelu/functional_1/stream_4/conv2d_4/BiasAdd:output:0*
T0*'
_output_shapes
:2%
#functional_1/stream_4/conv2d_4/ReluÔ
$functional_1/stream_5/ReadVariableOpReadVariableOp>functional_1_stream_5_readvariableop_streaming_stream_5_states*'
_output_shapes
:*
dtype02&
$functional_1/stream_5/ReadVariableOp«
)functional_1/stream_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2+
)functional_1/stream_5/strided_slice/stack¯
+functional_1/stream_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2-
+functional_1/stream_5/strided_slice/stack_1¯
+functional_1/stream_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2-
+functional_1/stream_5/strided_slice/stack_2
#functional_1/stream_5/strided_sliceStridedSlice,functional_1/stream_5/ReadVariableOp:value:02functional_1/stream_5/strided_slice/stack:output:04functional_1/stream_5/strided_slice/stack_1:output:04functional_1/stream_5/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:*

begin_mask*
end_mask2%
#functional_1/stream_5/strided_slice
!functional_1/stream_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!functional_1/stream_5/concat/axis
functional_1/stream_5/concatConcatV2,functional_1/stream_5/strided_slice:output:01functional_1/stream_4/conv2d_4/Relu:activations:0*functional_1/stream_5/concat/axis:output:0*
N*
T0*'
_output_shapes
:2
functional_1/stream_5/concat
&functional_1/stream_5/AssignVariableOpAssignVariableOp>functional_1_stream_5_readvariableop_streaming_stream_5_states%functional_1/stream_5/concat:output:0%^functional_1/stream_5/ReadVariableOp*
_output_shapes
 *
dtype02(
&functional_1/stream_5/AssignVariableOp¬
4functional_1/stream_5/conv2d_5/Conv2D/ReadVariableOpReadVariableOpMfunctional_1_stream_5_conv2d_5_conv2d_readvariableop_stream_5_conv2d_5_kernel'^functional_1/stream_5/AssignVariableOp*'
_output_shapes
:@*
dtype026
4functional_1/stream_5/conv2d_5/Conv2D/ReadVariableOp
%functional_1/stream_5/conv2d_5/Conv2DConv2D%functional_1/stream_5/concat:output:0<functional_1/stream_5/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2'
%functional_1/stream_5/conv2d_5/Conv2D 
5functional_1/stream_5/conv2d_5/BiasAdd/ReadVariableOpReadVariableOpLfunctional_1_stream_5_conv2d_5_biasadd_readvariableop_stream_5_conv2d_5_bias'^functional_1/stream_5/AssignVariableOp*
_output_shapes
:@*
dtype027
5functional_1/stream_5/conv2d_5/BiasAdd/ReadVariableOpû
&functional_1/stream_5/conv2d_5/BiasAddBiasAdd.functional_1/stream_5/conv2d_5/Conv2D:output:0=functional_1/stream_5/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2(
&functional_1/stream_5/conv2d_5/BiasAdd´
#functional_1/stream_5/conv2d_5/ReluRelu/functional_1/stream_5/conv2d_5/BiasAdd:output:0*
T0*&
_output_shapes
:@2%
#functional_1/stream_5/conv2d_5/ReluÓ
$functional_1/stream_6/ReadVariableOpReadVariableOp>functional_1_stream_6_readvariableop_streaming_stream_6_states*&
_output_shapes
:@*
dtype02&
$functional_1/stream_6/ReadVariableOp«
)functional_1/stream_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2+
)functional_1/stream_6/strided_slice/stack¯
+functional_1/stream_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2-
+functional_1/stream_6/strided_slice/stack_1¯
+functional_1/stream_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2-
+functional_1/stream_6/strided_slice/stack_2
#functional_1/stream_6/strided_sliceStridedSlice,functional_1/stream_6/ReadVariableOp:value:02functional_1/stream_6/strided_slice/stack:output:04functional_1/stream_6/strided_slice/stack_1:output:04functional_1/stream_6/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:@*

begin_mask*
end_mask2%
#functional_1/stream_6/strided_slice
!functional_1/stream_6/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!functional_1/stream_6/concat/axis
functional_1/stream_6/concatConcatV2,functional_1/stream_6/strided_slice:output:01functional_1/stream_5/conv2d_5/Relu:activations:0*functional_1/stream_6/concat/axis:output:0*
N*
T0*&
_output_shapes
:@2
functional_1/stream_6/concat
&functional_1/stream_6/AssignVariableOpAssignVariableOp>functional_1_stream_6_readvariableop_streaming_stream_6_states%functional_1/stream_6/concat:output:0%^functional_1/stream_6/ReadVariableOp*
_output_shapes
 *
dtype02(
&functional_1/stream_6/AssignVariableOp¬
4functional_1/stream_6/conv2d_6/Conv2D/ReadVariableOpReadVariableOpMfunctional_1_stream_6_conv2d_6_conv2d_readvariableop_stream_6_conv2d_6_kernel'^functional_1/stream_6/AssignVariableOp*'
_output_shapes
:@*
dtype026
4functional_1/stream_6/conv2d_6/Conv2D/ReadVariableOp
%functional_1/stream_6/conv2d_6/Conv2DConv2D%functional_1/stream_6/concat:output:0<functional_1/stream_6/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:*
paddingVALID*
strides
2'
%functional_1/stream_6/conv2d_6/Conv2D¡
5functional_1/stream_6/conv2d_6/BiasAdd/ReadVariableOpReadVariableOpLfunctional_1_stream_6_conv2d_6_biasadd_readvariableop_stream_6_conv2d_6_bias'^functional_1/stream_6/AssignVariableOp*
_output_shapes	
:*
dtype027
5functional_1/stream_6/conv2d_6/BiasAdd/ReadVariableOpü
&functional_1/stream_6/conv2d_6/BiasAddBiasAdd.functional_1/stream_6/conv2d_6/Conv2D:output:0=functional_1/stream_6/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:2(
&functional_1/stream_6/conv2d_6/BiasAddµ
#functional_1/stream_6/conv2d_6/ReluRelu/functional_1/stream_6/conv2d_6/BiasAdd:output:0*
T0*'
_output_shapes
:2%
#functional_1/stream_6/conv2d_6/ReluÔ
$functional_1/stream_7/ReadVariableOpReadVariableOp>functional_1_stream_7_readvariableop_streaming_stream_7_states*'
_output_shapes
:*
dtype02&
$functional_1/stream_7/ReadVariableOp«
)functional_1/stream_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2+
)functional_1/stream_7/strided_slice/stack¯
+functional_1/stream_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2-
+functional_1/stream_7/strided_slice/stack_1¯
+functional_1/stream_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2-
+functional_1/stream_7/strided_slice/stack_2
#functional_1/stream_7/strided_sliceStridedSlice,functional_1/stream_7/ReadVariableOp:value:02functional_1/stream_7/strided_slice/stack:output:04functional_1/stream_7/strided_slice/stack_1:output:04functional_1/stream_7/strided_slice/stack_2:output:0*
Index0*
T0*%
_output_shapes
: *

begin_mask*
end_mask2%
#functional_1/stream_7/strided_slice
!functional_1/stream_7/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!functional_1/stream_7/concat/axis
functional_1/stream_7/concatConcatV2,functional_1/stream_7/strided_slice:output:01functional_1/stream_6/conv2d_6/Relu:activations:0*functional_1/stream_7/concat/axis:output:0*
N*
T0*'
_output_shapes
:2
functional_1/stream_7/concat
&functional_1/stream_7/AssignVariableOpAssignVariableOp>functional_1_stream_7_readvariableop_streaming_stream_7_states%functional_1/stream_7/concat:output:0%^functional_1/stream_7/ReadVariableOp*
_output_shapes
 *
dtype02(
&functional_1/stream_7/AssignVariableOpÄ
#functional_1/stream_7/flatten/ConstConst'^functional_1/stream_7/AssignVariableOp*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2%
#functional_1/stream_7/flatten/ConstØ
%functional_1/stream_7/flatten/ReshapeReshape%functional_1/stream_7/concat:output:0,functional_1/stream_7/flatten/Const:output:0*
T0*
_output_shapes
:	2'
%functional_1/stream_7/flatten/Reshape¤
functional_1/dropout/IdentityIdentity.functional_1/stream_7/flatten/Reshape:output:0*
T0*
_output_shapes
:	2
functional_1/dropout/IdentityÖ
(functional_1/dense/MatMul/ReadVariableOpReadVariableOp?functional_1_dense_matmul_readvariableop_streaming_dense_kernel* 
_output_shapes
:
*
dtype02*
(functional_1/dense/MatMul/ReadVariableOpÄ
functional_1/dense/MatMulMatMul&functional_1/dropout/Identity:output:00functional_1/dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
functional_1/dense/MatMulÒ
)functional_1/dense/BiasAdd/ReadVariableOpReadVariableOp>functional_1_dense_biasadd_readvariableop_streaming_dense_bias*
_output_shapes	
:*
dtype02+
)functional_1/dense/BiasAdd/ReadVariableOpÅ
functional_1/dense/BiasAddBiasAdd#functional_1/dense/MatMul:product:01functional_1/dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
functional_1/dense/BiasAddÞ
*functional_1/dense_1/MatMul/ReadVariableOpReadVariableOpCfunctional_1_dense_1_matmul_readvariableop_streaming_dense_1_kernel* 
_output_shapes
:
*
dtype02,
*functional_1/dense_1/MatMul/ReadVariableOpÇ
functional_1/dense_1/MatMulMatMul#functional_1/dense/BiasAdd:output:02functional_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
functional_1/dense_1/MatMulÚ
+functional_1/dense_1/BiasAdd/ReadVariableOpReadVariableOpBfunctional_1_dense_1_biasadd_readvariableop_streaming_dense_1_bias*
_output_shapes	
:*
dtype02-
+functional_1/dense_1/BiasAdd/ReadVariableOpÍ
functional_1/dense_1/BiasAddBiasAdd%functional_1/dense_1/MatMul:product:03functional_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
functional_1/dense_1/BiasAdd
functional_1/dense_1/ReluRelu%functional_1/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:	2
functional_1/dense_1/ReluÝ
*functional_1/dense_2/MatMul/ReadVariableOpReadVariableOpCfunctional_1_dense_2_matmul_readvariableop_streaming_dense_2_kernel*
_output_shapes
:	*
dtype02,
*functional_1/dense_2/MatMul/ReadVariableOpÊ
functional_1/dense_2/MatMulMatMul'functional_1/dense_1/Relu:activations:02functional_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/dense_2/MatMulÙ
+functional_1/dense_2/BiasAdd/ReadVariableOpReadVariableOpBfunctional_1_dense_2_biasadd_readvariableop_streaming_dense_2_bias*
_output_shapes
:*
dtype02-
+functional_1/dense_2/BiasAdd/ReadVariableOpÌ
functional_1/dense_2/BiasAddBiasAdd%functional_1/dense_2/MatMul:product:03functional_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/dense_2/BiasAdd¶
IdentityIdentity%functional_1/dense_2/BiasAdd:output:0%^functional_1/stream/AssignVariableOp'^functional_1/stream_1/AssignVariableOp'^functional_1/stream_2/AssignVariableOp'^functional_1/stream_3/AssignVariableOp'^functional_1/stream_4/AssignVariableOp'^functional_1/stream_5/AssignVariableOp'^functional_1/stream_6/AssignVariableOp'^functional_1/stream_7/AssignVariableOp*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*
_input_shapes
~:::::::::::::::::::::::::::::2L
$functional_1/stream/AssignVariableOp$functional_1/stream/AssignVariableOp2P
&functional_1/stream_1/AssignVariableOp&functional_1/stream_1/AssignVariableOp2P
&functional_1/stream_2/AssignVariableOp&functional_1/stream_2/AssignVariableOp2P
&functional_1/stream_3/AssignVariableOp&functional_1/stream_3/AssignVariableOp2P
&functional_1/stream_4/AssignVariableOp&functional_1/stream_4/AssignVariableOp2P
&functional_1/stream_5/AssignVariableOp&functional_1/stream_5/AssignVariableOp2P
&functional_1/stream_6/AssignVariableOp&functional_1/stream_6/AssignVariableOp2P
&functional_1/stream_7/AssignVariableOp&functional_1/stream_7/AssignVariableOp:X T
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinput_audio:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Æ
¢
B__inference_stream_7_layer_call_and_return_conditional_losses_3078

inputs,
(readvariableop_streaming_stream_7_states
identity¢AssignVariableOp
ReadVariableOpReadVariableOp(readvariableop_streaming_stream_7_states*'
_output_shapes
:*
dtype02
ReadVariableOp
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*%
_output_shapes
: *

begin_mask*
end_mask2
strided_slice\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2strided_slice:output:0inputsconcat/axis:output:0*
N*
T0*'
_output_shapes
:2
concat§
AssignVariableOpAssignVariableOp(readvariableop_streaming_stream_7_statesconcat:output:0^ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOp
flatten/ConstConst^AssignVariableOp*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
flatten/Const
flatten/ReshapeReshapeconcat:output:0flatten/Const:output:0*
T0*
_output_shapes
:	2
flatten/Reshapew
IdentityIdentityflatten/Reshape:output:0^AssignVariableOp*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0**
_input_shapes
::2$
AssignVariableOpAssignVariableOp:O K
'
_output_shapes
:
 
_user_specified_nameinputs:

_output_shapes
: 

Q
5__inference_tf_op_layer_ExpandDims_layer_call_fn_4144

inputs
identityÊ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*Y
fTRR
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_28182
PartitionedCallk
IdentityIdentityPartitionedCall:output:0*
T0*&
_output_shapes
:2

Identity"
identityIdentity:output:0*!
_input_shapes
::J F
"
_output_shapes
:
 
_user_specified_nameinputs
Û
l
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_2818

inputs
identityk
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
ExpandDims/dim

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*
_cloned(*&
_output_shapes
:2

ExpandDimsf
IdentityIdentityExpandDims:output:0*
T0*&
_output_shapes
:2

Identity"
identityIdentity:output:0*!
_input_shapes
::J F
"
_output_shapes
:
 
_user_specified_nameinputs
ð
Ç
A__inference_dense_1_layer_call_and_return_conditional_losses_3151

inputs2
.matmul_readvariableop_streaming_dense_1_kernel1
-biasadd_readvariableop_streaming_dense_1_bias
identity
MatMul/ReadVariableOpReadVariableOp.matmul_readvariableop_streaming_dense_1_kernel* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpk
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
MatMul
BiasAdd/ReadVariableOpReadVariableOp-biasadd_readvariableop_streaming_dense_1_bias*
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
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 


B__inference_stream_3_layer_call_and_return_conditional_losses_2948

inputs,
(readvariableop_streaming_stream_3_states;
7conv2d_3_conv2d_readvariableop_stream_3_conv2d_3_kernel:
6conv2d_3_biasadd_readvariableop_stream_3_conv2d_3_bias
identity¢AssignVariableOp
ReadVariableOpReadVariableOp(readvariableop_streaming_stream_3_states*&
_output_shapes
:@*
dtype02
ReadVariableOp
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:@*

begin_mask*
end_mask2
strided_slice\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2strided_slice:output:0inputsconcat/axis:output:0*
N*
T0*&
_output_shapes
:@2
concat§
AssignVariableOpAssignVariableOp(readvariableop_streaming_stream_3_statesconcat:output:0^ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOp
conv2d_3/dilation_rateConst^AssignVariableOp*
_output_shapes
:*
dtype0*
valueB"      2
conv2d_3/dilation_rateÓ
conv2d_3/Conv2D/ReadVariableOpReadVariableOp7conv2d_3_conv2d_readvariableop_stream_3_conv2d_3_kernel^AssignVariableOp*&
_output_shapes
:@@*
dtype02 
conv2d_3/Conv2D/ReadVariableOp¿
conv2d_3/Conv2DConv2Dconcat:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
conv2d_3/Conv2DÈ
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp6conv2d_3_biasadd_readvariableop_stream_3_conv2d_3_bias^AssignVariableOp*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp£
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
conv2d_3/BiasAddr
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*&
_output_shapes
:@2
conv2d_3/Relu
IdentityIdentityconv2d_3/Relu:activations:0^AssignVariableOp*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*1
_input_shapes 
:@:::2$
AssignVariableOpAssignVariableOp:N J
&
_output_shapes
:@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ä
È
B__inference_conv2d_4_layer_call_and_return_conditional_losses_2737

inputs2
.conv2d_readvariableop_stream_4_conv2d_4_kernel1
-biasadd_readvariableop_stream_4_conv2d_4_bias
identityo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate¦
Conv2D/ReadVariableOpReadVariableOp.conv2d_readvariableop_stream_4_conv2d_4_kernel*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp·
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOp-biasadd_readvariableop_stream_4_conv2d_4_bias*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
¬

&__inference_dense_1_layer_call_fn_4415

inputs
streaming_dense_1_kernel
streaming_dense_1_bias
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstreaming_dense_1_kernelstreaming_dense_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_31512
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
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Þ
È
B__inference_conv2d_2_layer_call_and_return_conditional_losses_2671

inputs2
.conv2d_readvariableop_stream_2_conv2d_2_kernel1
-biasadd_readvariableop_stream_2_conv2d_2_bias
identityo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate¥
Conv2D/ReadVariableOpReadVariableOp.conv2d_readvariableop_stream_2_conv2d_2_kernel*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¶
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOp-biasadd_readvariableop_stream_2_conv2d_2_bias*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2
Relu
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
÷N

F__inference_functional_1_layer_call_and_return_conditional_losses_3279

inputs"
stream_streaming_stream_states
stream_stream_conv2d_kernel
stream_stream_conv2d_bias&
"stream_1_streaming_stream_1_states%
!stream_1_stream_1_conv2d_1_kernel#
stream_1_stream_1_conv2d_1_bias&
"stream_2_streaming_stream_2_states%
!stream_2_stream_2_conv2d_2_kernel#
stream_2_stream_2_conv2d_2_bias&
"stream_3_streaming_stream_3_states%
!stream_3_stream_3_conv2d_3_kernel#
stream_3_stream_3_conv2d_3_bias&
"stream_4_streaming_stream_4_states%
!stream_4_stream_4_conv2d_4_kernel#
stream_4_stream_4_conv2d_4_bias&
"stream_5_streaming_stream_5_states%
!stream_5_stream_5_conv2d_5_kernel#
stream_5_stream_5_conv2d_5_bias&
"stream_6_streaming_stream_6_states%
!stream_6_stream_6_conv2d_6_kernel#
stream_6_stream_6_conv2d_6_bias&
"stream_7_streaming_stream_7_states 
dense_streaming_dense_kernel
dense_streaming_dense_bias$
 dense_1_streaming_dense_1_kernel"
dense_1_streaming_dense_1_bias$
 dense_2_streaming_dense_2_kernel"
dense_2_streaming_dense_2_bias
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dropout/StatefulPartitionedCall¢stream/StatefulPartitionedCall¢ stream_1/StatefulPartitionedCall¢ stream_2/StatefulPartitionedCall¢ stream_3/StatefulPartitionedCall¢ stream_4/StatefulPartitionedCall¢ stream_5/StatefulPartitionedCall¢ stream_6/StatefulPartitionedCall¢ stream_7/StatefulPartitionedCallø
&tf_op_layer_ExpandDims/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*Y
fTRR
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_28182(
&tf_op_layer_ExpandDims/PartitionedCallè
stream/StatefulPartitionedCallStatefulPartitionedCall/tf_op_layer_ExpandDims/PartitionedCall:output:0stream_streaming_stream_statesstream_stream_conv2d_kernelstream_stream_conv2d_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_stream_layer_call_and_return_conditional_losses_28462 
stream/StatefulPartitionedCallö
 stream_1/StatefulPartitionedCallStatefulPartitionedCall'stream/StatefulPartitionedCall:output:0"stream_1_streaming_stream_1_states!stream_1_stream_1_conv2d_1_kernelstream_1_stream_1_conv2d_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_stream_1_layer_call_and_return_conditional_losses_28802"
 stream_1/StatefulPartitionedCallø
 stream_2/StatefulPartitionedCallStatefulPartitionedCall)stream_1/StatefulPartitionedCall:output:0"stream_2_streaming_stream_2_states!stream_2_stream_2_conv2d_2_kernelstream_2_stream_2_conv2d_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_stream_2_layer_call_and_return_conditional_losses_29142"
 stream_2/StatefulPartitionedCallø
 stream_3/StatefulPartitionedCallStatefulPartitionedCall)stream_2/StatefulPartitionedCall:output:0"stream_3_streaming_stream_3_states!stream_3_stream_3_conv2d_3_kernelstream_3_stream_3_conv2d_3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_stream_3_layer_call_and_return_conditional_losses_29482"
 stream_3/StatefulPartitionedCallù
 stream_4/StatefulPartitionedCallStatefulPartitionedCall)stream_3/StatefulPartitionedCall:output:0"stream_4_streaming_stream_4_states!stream_4_stream_4_conv2d_4_kernelstream_4_stream_4_conv2d_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_stream_4_layer_call_and_return_conditional_losses_29822"
 stream_4/StatefulPartitionedCallø
 stream_5/StatefulPartitionedCallStatefulPartitionedCall)stream_4/StatefulPartitionedCall:output:0"stream_5_streaming_stream_5_states!stream_5_stream_5_conv2d_5_kernelstream_5_stream_5_conv2d_5_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_stream_5_layer_call_and_return_conditional_losses_30162"
 stream_5/StatefulPartitionedCallù
 stream_6/StatefulPartitionedCallStatefulPartitionedCall)stream_5/StatefulPartitionedCall:output:0"stream_6_streaming_stream_6_states!stream_6_stream_6_conv2d_6_kernelstream_6_stream_6_conv2d_6_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_stream_6_layer_call_and_return_conditional_losses_30502"
 stream_6/StatefulPartitionedCall§
 stream_7/StatefulPartitionedCallStatefulPartitionedCall)stream_6/StatefulPartitionedCall:output:0"stream_7_streaming_stream_7_states*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_stream_7_layer_call_and_return_conditional_losses_30782"
 stream_7/StatefulPartitionedCallÿ
dropout/StatefulPartitionedCallStatefulPartitionedCall)stream_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_31002!
dropout/StatefulPartitionedCall¸
dense/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_streaming_dense_kerneldense_streaming_dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_31282
dense/StatefulPartitionedCallÄ
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0 dense_1_streaming_dense_1_kerneldense_1_streaming_dense_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_31512!
dense_1/StatefulPartitionedCallÅ
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0 dense_2_streaming_dense_2_kerneldense_2_streaming_dense_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_31732!
dense_2/StatefulPartitionedCall
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall^stream/StatefulPartitionedCall!^stream_1/StatefulPartitionedCall!^stream_2/StatefulPartitionedCall!^stream_3/StatefulPartitionedCall!^stream_4/StatefulPartitionedCall!^stream_5/StatefulPartitionedCall!^stream_6/StatefulPartitionedCall!^stream_7/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*
_input_shapes
~:::::::::::::::::::::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2@
stream/StatefulPartitionedCallstream/StatefulPartitionedCall2D
 stream_1/StatefulPartitionedCall stream_1/StatefulPartitionedCall2D
 stream_2/StatefulPartitionedCall stream_2/StatefulPartitionedCall2D
 stream_3/StatefulPartitionedCall stream_3/StatefulPartitionedCall2D
 stream_4/StatefulPartitionedCall stream_4/StatefulPartitionedCall2D
 stream_5/StatefulPartitionedCall stream_5/StatefulPartitionedCall2D
 stream_6/StatefulPartitionedCall stream_6/StatefulPartitionedCall2D
 stream_7/StatefulPartitionedCall stream_7/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ä
È
B__inference_conv2d_6_layer_call_and_return_conditional_losses_2788

inputs2
.conv2d_readvariableop_stream_6_conv2d_6_kernel1
-biasadd_readvariableop_stream_6_conv2d_6_bias
identityo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate¦
Conv2D/ReadVariableOpReadVariableOp.conv2d_readvariableop_stream_6_conv2d_6_kernel*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp·
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOp-biasadd_readvariableop_stream_6_conv2d_6_bias*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
á
È
B__inference_conv2d_5_layer_call_and_return_conditional_losses_2770

inputs2
.conv2d_readvariableop_stream_5_conv2d_5_kernel1
-biasadd_readvariableop_stream_5_conv2d_5_bias
identityo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate¦
Conv2D/ReadVariableOpReadVariableOp.conv2d_readvariableop_stream_5_conv2d_5_kernel*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp¶
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOp-biasadd_readvariableop_stream_5_conv2d_5_bias*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2
Relu
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
§
¹
'__inference_stream_1_layer_call_fn_4198

inputs
streaming_stream_1_states
stream_1_conv2d_1_kernel
stream_1_conv2d_1_bias
identity¢StatefulPartitionedCall¨
StatefulPartitionedCallStatefulPartitionedCallinputsstreaming_stream_1_statesstream_1_conv2d_1_kernelstream_1_conv2d_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_stream_1_layer_call_and_return_conditional_losses_28802
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*1
_input_shapes 
:@:::22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ø

`
A__inference_dropout_layer_call_and_return_conditional_losses_4365

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constk
dropout/MulMulinputsdropout/Const:output:0*
T0*
_output_shapes
:	2
dropout/Mulo
dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     2
dropout/Shape¬
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
_output_shapes
:	*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y¶
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	2
dropout/GreaterEqualw
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	2
dropout/Castr
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*
_output_shapes
:	2
dropout/Mul_1]
IdentityIdentitydropout/Mul_1:z:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs
Þ
È
B__inference_conv2d_1_layer_call_and_return_conditional_losses_2623

inputs2
.conv2d_readvariableop_stream_1_conv2d_1_kernel1
-biasadd_readvariableop_stream_1_conv2d_1_bias
identityo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate¥
Conv2D/ReadVariableOpReadVariableOp.conv2d_readvariableop_stream_1_conv2d_1_kernel*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¶
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOp-biasadd_readvariableop_stream_1_conv2d_1_bias*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2
Relu
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 

­
%__inference_stream_layer_call_fn_4171

inputs
streaming_stream_states
stream_conv2d_kernel
stream_conv2d_bias
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstreaming_stream_statesstream_conv2d_kernelstream_conv2d_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_stream_layer_call_and_return_conditional_losses_28462
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*1
_input_shapes 
::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 


B__inference_stream_1_layer_call_and_return_conditional_losses_2880

inputs,
(readvariableop_streaming_stream_1_states;
7conv2d_1_conv2d_readvariableop_stream_1_conv2d_1_kernel:
6conv2d_1_biasadd_readvariableop_stream_1_conv2d_1_bias
identity¢AssignVariableOp
ReadVariableOpReadVariableOp(readvariableop_streaming_stream_1_states*&
_output_shapes
:@*
dtype02
ReadVariableOp
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:@*

begin_mask*
end_mask2
strided_slice\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2strided_slice:output:0inputsconcat/axis:output:0*
N*
T0*&
_output_shapes
:@2
concat§
AssignVariableOpAssignVariableOp(readvariableop_streaming_stream_1_statesconcat:output:0^ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOp
conv2d_1/dilation_rateConst^AssignVariableOp*
_output_shapes
:*
dtype0*
valueB"      2
conv2d_1/dilation_rateÓ
conv2d_1/Conv2D/ReadVariableOpReadVariableOp7conv2d_1_conv2d_readvariableop_stream_1_conv2d_1_kernel^AssignVariableOp*&
_output_shapes
:@@*
dtype02 
conv2d_1/Conv2D/ReadVariableOp¿
conv2d_1/Conv2DConv2Dconcat:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
conv2d_1/Conv2DÈ
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp6conv2d_1_biasadd_readvariableop_stream_1_conv2d_1_bias^AssignVariableOp*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp£
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
conv2d_1/BiasAddr
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*&
_output_shapes
:@2
conv2d_1/Relu
IdentityIdentityconv2d_1/Relu:activations:0^AssignVariableOp*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*1
_input_shapes 
:@:::2$
AssignVariableOpAssignVariableOp:N J
&
_output_shapes
:@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
õ
_
&__inference_dropout_layer_call_fn_4375

inputs
identity¢StatefulPartitionedCallÌ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_31002
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*
_input_shapes
:	22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	
 
_user_specified_nameinputs
ð
Ç
A__inference_dense_1_layer_call_and_return_conditional_losses_4408

inputs2
.matmul_readvariableop_streaming_dense_1_kernel1
-biasadd_readvariableop_streaming_dense_1_bias
identity
MatMul/ReadVariableOpReadVariableOp.matmul_readvariableop_streaming_dense_1_kernel* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpk
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
MatMul
BiasAdd/ReadVariableOpReadVariableOp-biasadd_readvariableop_streaming_dense_1_bias*
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
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
þ

+__inference_functional_1_layer_call_fn_4100
input_audio
streaming_stream_states
stream_conv2d_kernel
stream_conv2d_bias
streaming_stream_1_states
stream_1_conv2d_1_kernel
stream_1_conv2d_1_bias
streaming_stream_2_states
stream_2_conv2d_2_kernel
stream_2_conv2d_2_bias
streaming_stream_3_states
stream_3_conv2d_3_kernel
stream_3_conv2d_3_bias
streaming_stream_4_states
stream_4_conv2d_4_kernel
stream_4_conv2d_4_bias
streaming_stream_5_states
stream_5_conv2d_5_kernel
stream_5_conv2d_5_bias
streaming_stream_6_states
stream_6_conv2d_6_kernel
stream_6_conv2d_6_bias
streaming_stream_7_states
streaming_dense_kernel
streaming_dense_bias
streaming_dense_1_kernel
streaming_dense_1_bias
streaming_dense_2_kernel
streaming_dense_2_bias
identity¢StatefulPartitionedCallÅ
StatefulPartitionedCallStatefulPartitionedCallinput_audiostreaming_stream_statesstream_conv2d_kernelstream_conv2d_biasstreaming_stream_1_statesstream_1_conv2d_1_kernelstream_1_conv2d_1_biasstreaming_stream_2_statesstream_2_conv2d_2_kernelstream_2_conv2d_2_biasstreaming_stream_3_statesstream_3_conv2d_3_kernelstream_3_conv2d_3_biasstreaming_stream_4_statesstream_4_conv2d_4_kernelstream_4_conv2d_4_biasstreaming_stream_5_statesstream_5_conv2d_5_kernelstream_5_conv2d_5_biasstreaming_stream_6_statesstream_6_conv2d_6_kernelstream_6_conv2d_6_biasstreaming_stream_7_statesstreaming_dense_kernelstreaming_dense_biasstreaming_dense_1_kernelstreaming_dense_1_biasstreaming_dense_2_kernelstreaming_dense_2_bias*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*6
_read_only_resource_inputs
	**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_functional_1_layer_call_and_return_conditional_losses_32792
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinput_audio:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ï

+__inference_functional_1_layer_call_fn_3778

inputs
streaming_stream_states
stream_conv2d_kernel
stream_conv2d_bias
streaming_stream_1_states
stream_1_conv2d_1_kernel
stream_1_conv2d_1_bias
streaming_stream_2_states
stream_2_conv2d_2_kernel
stream_2_conv2d_2_bias
streaming_stream_3_states
stream_3_conv2d_3_kernel
stream_3_conv2d_3_bias
streaming_stream_4_states
stream_4_conv2d_4_kernel
stream_4_conv2d_4_bias
streaming_stream_5_states
stream_5_conv2d_5_kernel
stream_5_conv2d_5_bias
streaming_stream_6_states
stream_6_conv2d_6_kernel
stream_6_conv2d_6_bias
streaming_stream_7_states
streaming_dense_kernel
streaming_dense_bias
streaming_dense_1_kernel
streaming_dense_1_bias
streaming_dense_2_kernel
streaming_dense_2_bias
identity¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCallinputsstreaming_stream_statesstream_conv2d_kernelstream_conv2d_biasstreaming_stream_1_statesstream_1_conv2d_1_kernelstream_1_conv2d_1_biasstreaming_stream_2_statesstream_2_conv2d_2_kernelstream_2_conv2d_2_biasstreaming_stream_3_statesstream_3_conv2d_3_kernelstream_3_conv2d_3_biasstreaming_stream_4_statesstream_4_conv2d_4_kernelstream_4_conv2d_4_biasstreaming_stream_5_statesstream_5_conv2d_5_kernelstream_5_conv2d_5_biasstreaming_stream_6_statesstream_6_conv2d_6_kernelstream_6_conv2d_6_biasstreaming_stream_7_statesstreaming_dense_kernelstreaming_dense_biasstreaming_dense_1_kernelstreaming_dense_1_biasstreaming_dense_2_kernelstreaming_dense_2_bias*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*6
_read_only_resource_inputs
	**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_functional_1_layer_call_and_return_conditional_losses_33572
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
µË

F__inference_functional_1_layer_call_and_return_conditional_losses_3712

inputs1
-stream_readvariableop_streaming_stream_states<
8stream_conv2d_conv2d_readvariableop_stream_conv2d_kernel;
7stream_conv2d_biasadd_readvariableop_stream_conv2d_bias5
1stream_1_readvariableop_streaming_stream_1_statesD
@stream_1_conv2d_1_conv2d_readvariableop_stream_1_conv2d_1_kernelC
?stream_1_conv2d_1_biasadd_readvariableop_stream_1_conv2d_1_bias5
1stream_2_readvariableop_streaming_stream_2_statesD
@stream_2_conv2d_2_conv2d_readvariableop_stream_2_conv2d_2_kernelC
?stream_2_conv2d_2_biasadd_readvariableop_stream_2_conv2d_2_bias5
1stream_3_readvariableop_streaming_stream_3_statesD
@stream_3_conv2d_3_conv2d_readvariableop_stream_3_conv2d_3_kernelC
?stream_3_conv2d_3_biasadd_readvariableop_stream_3_conv2d_3_bias5
1stream_4_readvariableop_streaming_stream_4_statesD
@stream_4_conv2d_4_conv2d_readvariableop_stream_4_conv2d_4_kernelC
?stream_4_conv2d_4_biasadd_readvariableop_stream_4_conv2d_4_bias5
1stream_5_readvariableop_streaming_stream_5_statesD
@stream_5_conv2d_5_conv2d_readvariableop_stream_5_conv2d_5_kernelC
?stream_5_conv2d_5_biasadd_readvariableop_stream_5_conv2d_5_bias5
1stream_6_readvariableop_streaming_stream_6_statesD
@stream_6_conv2d_6_conv2d_readvariableop_stream_6_conv2d_6_kernelC
?stream_6_conv2d_6_biasadd_readvariableop_stream_6_conv2d_6_bias5
1stream_7_readvariableop_streaming_stream_7_states6
2dense_matmul_readvariableop_streaming_dense_kernel5
1dense_biasadd_readvariableop_streaming_dense_bias:
6dense_1_matmul_readvariableop_streaming_dense_1_kernel9
5dense_1_biasadd_readvariableop_streaming_dense_1_bias:
6dense_2_matmul_readvariableop_streaming_dense_2_kernel9
5dense_2_biasadd_readvariableop_streaming_dense_2_bias
identity¢stream/AssignVariableOp¢stream_1/AssignVariableOp¢stream_2/AssignVariableOp¢stream_3/AssignVariableOp¢stream_4/AssignVariableOp¢stream_5/AssignVariableOp¢stream_6/AssignVariableOp¢stream_7/AssignVariableOp
%tf_op_layer_ExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2'
%tf_op_layer_ExpandDims/ExpandDims/dimÌ
!tf_op_layer_ExpandDims/ExpandDims
ExpandDimsinputs.tf_op_layer_ExpandDims/ExpandDims/dim:output:0*
T0*
_cloned(*&
_output_shapes
:2#
!tf_op_layer_ExpandDims/ExpandDims¤
stream/ReadVariableOpReadVariableOp-stream_readvariableop_streaming_stream_states*&
_output_shapes
:*
dtype02
stream/ReadVariableOp
stream/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
stream/strided_slice/stack
stream/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
stream/strided_slice/stack_1
stream/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
stream/strided_slice/stack_2®
stream/strided_sliceStridedSlicestream/ReadVariableOp:value:0#stream/strided_slice/stack:output:0%stream/strided_slice/stack_1:output:0%stream/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:*

begin_mask*
end_mask2
stream/strided_slicej
stream/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
stream/concat/axisÌ
stream/concatConcatV2stream/strided_slice:output:0*tf_op_layer_ExpandDims/ExpandDims:output:0stream/concat/axis:output:0*
N*
T0*&
_output_shapes
:2
stream/concatÈ
stream/AssignVariableOpAssignVariableOp-stream_readvariableop_streaming_stream_statesstream/concat:output:0^stream/ReadVariableOp*
_output_shapes
 *
dtype02
stream/AssignVariableOpå
#stream/conv2d/Conv2D/ReadVariableOpReadVariableOp8stream_conv2d_conv2d_readvariableop_stream_conv2d_kernel^stream/AssignVariableOp*&
_output_shapes
:@*
dtype02%
#stream/conv2d/Conv2D/ReadVariableOpÕ
stream/conv2d/Conv2DConv2Dstream/concat:output:0+stream/conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
stream/conv2d/Conv2DÚ
$stream/conv2d/BiasAdd/ReadVariableOpReadVariableOp7stream_conv2d_biasadd_readvariableop_stream_conv2d_bias^stream/AssignVariableOp*
_output_shapes
:@*
dtype02&
$stream/conv2d/BiasAdd/ReadVariableOp·
stream/conv2d/BiasAddBiasAddstream/conv2d/Conv2D:output:0,stream/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
stream/conv2d/BiasAdd
stream/conv2d/ReluRelustream/conv2d/BiasAdd:output:0*
T0*&
_output_shapes
:@2
stream/conv2d/Relu¬
stream_1/ReadVariableOpReadVariableOp1stream_1_readvariableop_streaming_stream_1_states*&
_output_shapes
:@*
dtype02
stream_1/ReadVariableOp
stream_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
stream_1/strided_slice/stack
stream_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2 
stream_1/strided_slice/stack_1
stream_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2 
stream_1/strided_slice/stack_2º
stream_1/strided_sliceStridedSlicestream_1/ReadVariableOp:value:0%stream_1/strided_slice/stack:output:0'stream_1/strided_slice/stack_1:output:0'stream_1/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:@*

begin_mask*
end_mask2
stream_1/strided_slicen
stream_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
stream_1/concat/axisÊ
stream_1/concatConcatV2stream_1/strided_slice:output:0 stream/conv2d/Relu:activations:0stream_1/concat/axis:output:0*
N*
T0*&
_output_shapes
:@2
stream_1/concatÔ
stream_1/AssignVariableOpAssignVariableOp1stream_1_readvariableop_streaming_stream_1_statesstream_1/concat:output:0^stream_1/ReadVariableOp*
_output_shapes
 *
dtype02
stream_1/AssignVariableOp÷
'stream_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp@stream_1_conv2d_1_conv2d_readvariableop_stream_1_conv2d_1_kernel^stream_1/AssignVariableOp*&
_output_shapes
:@@*
dtype02)
'stream_1/conv2d_1/Conv2D/ReadVariableOpã
stream_1/conv2d_1/Conv2DConv2Dstream_1/concat:output:0/stream_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
stream_1/conv2d_1/Conv2Dì
(stream_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp?stream_1_conv2d_1_biasadd_readvariableop_stream_1_conv2d_1_bias^stream_1/AssignVariableOp*
_output_shapes
:@*
dtype02*
(stream_1/conv2d_1/BiasAdd/ReadVariableOpÇ
stream_1/conv2d_1/BiasAddBiasAdd!stream_1/conv2d_1/Conv2D:output:00stream_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
stream_1/conv2d_1/BiasAdd
stream_1/conv2d_1/ReluRelu"stream_1/conv2d_1/BiasAdd:output:0*
T0*&
_output_shapes
:@2
stream_1/conv2d_1/Relu¬
stream_2/ReadVariableOpReadVariableOp1stream_2_readvariableop_streaming_stream_2_states*&
_output_shapes
:@*
dtype02
stream_2/ReadVariableOp
stream_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
stream_2/strided_slice/stack
stream_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2 
stream_2/strided_slice/stack_1
stream_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2 
stream_2/strided_slice/stack_2º
stream_2/strided_sliceStridedSlicestream_2/ReadVariableOp:value:0%stream_2/strided_slice/stack:output:0'stream_2/strided_slice/stack_1:output:0'stream_2/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:@*

begin_mask*
end_mask2
stream_2/strided_slicen
stream_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
stream_2/concat/axisÎ
stream_2/concatConcatV2stream_2/strided_slice:output:0$stream_1/conv2d_1/Relu:activations:0stream_2/concat/axis:output:0*
N*
T0*&
_output_shapes
:@2
stream_2/concatÔ
stream_2/AssignVariableOpAssignVariableOp1stream_2_readvariableop_streaming_stream_2_statesstream_2/concat:output:0^stream_2/ReadVariableOp*
_output_shapes
 *
dtype02
stream_2/AssignVariableOp÷
'stream_2/conv2d_2/Conv2D/ReadVariableOpReadVariableOp@stream_2_conv2d_2_conv2d_readvariableop_stream_2_conv2d_2_kernel^stream_2/AssignVariableOp*&
_output_shapes
:@@*
dtype02)
'stream_2/conv2d_2/Conv2D/ReadVariableOpã
stream_2/conv2d_2/Conv2DConv2Dstream_2/concat:output:0/stream_2/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
stream_2/conv2d_2/Conv2Dì
(stream_2/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp?stream_2_conv2d_2_biasadd_readvariableop_stream_2_conv2d_2_bias^stream_2/AssignVariableOp*
_output_shapes
:@*
dtype02*
(stream_2/conv2d_2/BiasAdd/ReadVariableOpÇ
stream_2/conv2d_2/BiasAddBiasAdd!stream_2/conv2d_2/Conv2D:output:00stream_2/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
stream_2/conv2d_2/BiasAdd
stream_2/conv2d_2/ReluRelu"stream_2/conv2d_2/BiasAdd:output:0*
T0*&
_output_shapes
:@2
stream_2/conv2d_2/Relu¬
stream_3/ReadVariableOpReadVariableOp1stream_3_readvariableop_streaming_stream_3_states*&
_output_shapes
:@*
dtype02
stream_3/ReadVariableOp
stream_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
stream_3/strided_slice/stack
stream_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2 
stream_3/strided_slice/stack_1
stream_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2 
stream_3/strided_slice/stack_2º
stream_3/strided_sliceStridedSlicestream_3/ReadVariableOp:value:0%stream_3/strided_slice/stack:output:0'stream_3/strided_slice/stack_1:output:0'stream_3/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:@*

begin_mask*
end_mask2
stream_3/strided_slicen
stream_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
stream_3/concat/axisÎ
stream_3/concatConcatV2stream_3/strided_slice:output:0$stream_2/conv2d_2/Relu:activations:0stream_3/concat/axis:output:0*
N*
T0*&
_output_shapes
:@2
stream_3/concatÔ
stream_3/AssignVariableOpAssignVariableOp1stream_3_readvariableop_streaming_stream_3_statesstream_3/concat:output:0^stream_3/ReadVariableOp*
_output_shapes
 *
dtype02
stream_3/AssignVariableOp÷
'stream_3/conv2d_3/Conv2D/ReadVariableOpReadVariableOp@stream_3_conv2d_3_conv2d_readvariableop_stream_3_conv2d_3_kernel^stream_3/AssignVariableOp*&
_output_shapes
:@@*
dtype02)
'stream_3/conv2d_3/Conv2D/ReadVariableOpã
stream_3/conv2d_3/Conv2DConv2Dstream_3/concat:output:0/stream_3/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
stream_3/conv2d_3/Conv2Dì
(stream_3/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp?stream_3_conv2d_3_biasadd_readvariableop_stream_3_conv2d_3_bias^stream_3/AssignVariableOp*
_output_shapes
:@*
dtype02*
(stream_3/conv2d_3/BiasAdd/ReadVariableOpÇ
stream_3/conv2d_3/BiasAddBiasAdd!stream_3/conv2d_3/Conv2D:output:00stream_3/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
stream_3/conv2d_3/BiasAdd
stream_3/conv2d_3/ReluRelu"stream_3/conv2d_3/BiasAdd:output:0*
T0*&
_output_shapes
:@2
stream_3/conv2d_3/Relu¬
stream_4/ReadVariableOpReadVariableOp1stream_4_readvariableop_streaming_stream_4_states*&
_output_shapes
:@*
dtype02
stream_4/ReadVariableOp
stream_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
stream_4/strided_slice/stack
stream_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2 
stream_4/strided_slice/stack_1
stream_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2 
stream_4/strided_slice/stack_2º
stream_4/strided_sliceStridedSlicestream_4/ReadVariableOp:value:0%stream_4/strided_slice/stack:output:0'stream_4/strided_slice/stack_1:output:0'stream_4/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:@*

begin_mask*
end_mask2
stream_4/strided_slicen
stream_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
stream_4/concat/axisÎ
stream_4/concatConcatV2stream_4/strided_slice:output:0$stream_3/conv2d_3/Relu:activations:0stream_4/concat/axis:output:0*
N*
T0*&
_output_shapes
:@2
stream_4/concatÔ
stream_4/AssignVariableOpAssignVariableOp1stream_4_readvariableop_streaming_stream_4_statesstream_4/concat:output:0^stream_4/ReadVariableOp*
_output_shapes
 *
dtype02
stream_4/AssignVariableOpø
'stream_4/conv2d_4/Conv2D/ReadVariableOpReadVariableOp@stream_4_conv2d_4_conv2d_readvariableop_stream_4_conv2d_4_kernel^stream_4/AssignVariableOp*'
_output_shapes
:@*
dtype02)
'stream_4/conv2d_4/Conv2D/ReadVariableOpä
stream_4/conv2d_4/Conv2DConv2Dstream_4/concat:output:0/stream_4/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:*
paddingVALID*
strides
2
stream_4/conv2d_4/Conv2Dí
(stream_4/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp?stream_4_conv2d_4_biasadd_readvariableop_stream_4_conv2d_4_bias^stream_4/AssignVariableOp*
_output_shapes	
:*
dtype02*
(stream_4/conv2d_4/BiasAdd/ReadVariableOpÈ
stream_4/conv2d_4/BiasAddBiasAdd!stream_4/conv2d_4/Conv2D:output:00stream_4/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:2
stream_4/conv2d_4/BiasAdd
stream_4/conv2d_4/ReluRelu"stream_4/conv2d_4/BiasAdd:output:0*
T0*'
_output_shapes
:2
stream_4/conv2d_4/Relu­
stream_5/ReadVariableOpReadVariableOp1stream_5_readvariableop_streaming_stream_5_states*'
_output_shapes
:*
dtype02
stream_5/ReadVariableOp
stream_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
stream_5/strided_slice/stack
stream_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2 
stream_5/strided_slice/stack_1
stream_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2 
stream_5/strided_slice/stack_2»
stream_5/strided_sliceStridedSlicestream_5/ReadVariableOp:value:0%stream_5/strided_slice/stack:output:0'stream_5/strided_slice/stack_1:output:0'stream_5/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:*

begin_mask*
end_mask2
stream_5/strided_slicen
stream_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
stream_5/concat/axisÏ
stream_5/concatConcatV2stream_5/strided_slice:output:0$stream_4/conv2d_4/Relu:activations:0stream_5/concat/axis:output:0*
N*
T0*'
_output_shapes
:2
stream_5/concatÔ
stream_5/AssignVariableOpAssignVariableOp1stream_5_readvariableop_streaming_stream_5_statesstream_5/concat:output:0^stream_5/ReadVariableOp*
_output_shapes
 *
dtype02
stream_5/AssignVariableOpø
'stream_5/conv2d_5/Conv2D/ReadVariableOpReadVariableOp@stream_5_conv2d_5_conv2d_readvariableop_stream_5_conv2d_5_kernel^stream_5/AssignVariableOp*'
_output_shapes
:@*
dtype02)
'stream_5/conv2d_5/Conv2D/ReadVariableOpã
stream_5/conv2d_5/Conv2DConv2Dstream_5/concat:output:0/stream_5/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
stream_5/conv2d_5/Conv2Dì
(stream_5/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp?stream_5_conv2d_5_biasadd_readvariableop_stream_5_conv2d_5_bias^stream_5/AssignVariableOp*
_output_shapes
:@*
dtype02*
(stream_5/conv2d_5/BiasAdd/ReadVariableOpÇ
stream_5/conv2d_5/BiasAddBiasAdd!stream_5/conv2d_5/Conv2D:output:00stream_5/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
stream_5/conv2d_5/BiasAdd
stream_5/conv2d_5/ReluRelu"stream_5/conv2d_5/BiasAdd:output:0*
T0*&
_output_shapes
:@2
stream_5/conv2d_5/Relu¬
stream_6/ReadVariableOpReadVariableOp1stream_6_readvariableop_streaming_stream_6_states*&
_output_shapes
:@*
dtype02
stream_6/ReadVariableOp
stream_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
stream_6/strided_slice/stack
stream_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2 
stream_6/strided_slice/stack_1
stream_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2 
stream_6/strided_slice/stack_2º
stream_6/strided_sliceStridedSlicestream_6/ReadVariableOp:value:0%stream_6/strided_slice/stack:output:0'stream_6/strided_slice/stack_1:output:0'stream_6/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:@*

begin_mask*
end_mask2
stream_6/strided_slicen
stream_6/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
stream_6/concat/axisÎ
stream_6/concatConcatV2stream_6/strided_slice:output:0$stream_5/conv2d_5/Relu:activations:0stream_6/concat/axis:output:0*
N*
T0*&
_output_shapes
:@2
stream_6/concatÔ
stream_6/AssignVariableOpAssignVariableOp1stream_6_readvariableop_streaming_stream_6_statesstream_6/concat:output:0^stream_6/ReadVariableOp*
_output_shapes
 *
dtype02
stream_6/AssignVariableOpø
'stream_6/conv2d_6/Conv2D/ReadVariableOpReadVariableOp@stream_6_conv2d_6_conv2d_readvariableop_stream_6_conv2d_6_kernel^stream_6/AssignVariableOp*'
_output_shapes
:@*
dtype02)
'stream_6/conv2d_6/Conv2D/ReadVariableOpä
stream_6/conv2d_6/Conv2DConv2Dstream_6/concat:output:0/stream_6/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:*
paddingVALID*
strides
2
stream_6/conv2d_6/Conv2Dí
(stream_6/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp?stream_6_conv2d_6_biasadd_readvariableop_stream_6_conv2d_6_bias^stream_6/AssignVariableOp*
_output_shapes	
:*
dtype02*
(stream_6/conv2d_6/BiasAdd/ReadVariableOpÈ
stream_6/conv2d_6/BiasAddBiasAdd!stream_6/conv2d_6/Conv2D:output:00stream_6/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:2
stream_6/conv2d_6/BiasAdd
stream_6/conv2d_6/ReluRelu"stream_6/conv2d_6/BiasAdd:output:0*
T0*'
_output_shapes
:2
stream_6/conv2d_6/Relu­
stream_7/ReadVariableOpReadVariableOp1stream_7_readvariableop_streaming_stream_7_states*'
_output_shapes
:*
dtype02
stream_7/ReadVariableOp
stream_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
stream_7/strided_slice/stack
stream_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2 
stream_7/strided_slice/stack_1
stream_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2 
stream_7/strided_slice/stack_2¹
stream_7/strided_sliceStridedSlicestream_7/ReadVariableOp:value:0%stream_7/strided_slice/stack:output:0'stream_7/strided_slice/stack_1:output:0'stream_7/strided_slice/stack_2:output:0*
Index0*
T0*%
_output_shapes
: *

begin_mask*
end_mask2
stream_7/strided_slicen
stream_7/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
stream_7/concat/axisÏ
stream_7/concatConcatV2stream_7/strided_slice:output:0$stream_6/conv2d_6/Relu:activations:0stream_7/concat/axis:output:0*
N*
T0*'
_output_shapes
:2
stream_7/concatÔ
stream_7/AssignVariableOpAssignVariableOp1stream_7_readvariableop_streaming_stream_7_statesstream_7/concat:output:0^stream_7/ReadVariableOp*
_output_shapes
 *
dtype02
stream_7/AssignVariableOp
stream_7/flatten/ConstConst^stream_7/AssignVariableOp*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
stream_7/flatten/Const¤
stream_7/flatten/ReshapeReshapestream_7/concat:output:0stream_7/flatten/Const:output:0*
T0*
_output_shapes
:	2
stream_7/flatten/Reshape}
dropout/IdentityIdentity!stream_7/flatten/Reshape:output:0*
T0*
_output_shapes
:	2
dropout/Identity¯
dense/MatMul/ReadVariableOpReadVariableOp2dense_matmul_readvariableop_streaming_dense_kernel* 
_output_shapes
:
*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMuldropout/Identity:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense/MatMul«
dense/BiasAdd/ReadVariableOpReadVariableOp1dense_biasadd_readvariableop_streaming_dense_bias*
_output_shapes	
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense/BiasAdd·
dense_1/MatMul/ReadVariableOpReadVariableOp6dense_1_matmul_readvariableop_streaming_dense_1_kernel* 
_output_shapes
:
*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldense/BiasAdd:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense_1/MatMul³
dense_1/BiasAdd/ReadVariableOpReadVariableOp5dense_1_biasadd_readvariableop_streaming_dense_1_bias*
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
dense_1/Relu¶
dense_2/MatMul/ReadVariableOpReadVariableOp6dense_2_matmul_readvariableop_streaming_dense_2_kernel*
_output_shapes
:	*
dtype02
dense_2/MatMul/ReadVariableOp
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_2/MatMul²
dense_2/BiasAdd/ReadVariableOpReadVariableOp5dense_2_biasadd_readvariableop_streaming_dense_2_bias*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_2/BiasAddÁ
IdentityIdentitydense_2/BiasAdd:output:0^stream/AssignVariableOp^stream_1/AssignVariableOp^stream_2/AssignVariableOp^stream_3/AssignVariableOp^stream_4/AssignVariableOp^stream_5/AssignVariableOp^stream_6/AssignVariableOp^stream_7/AssignVariableOp*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*
_input_shapes
~:::::::::::::::::::::::::::::22
stream/AssignVariableOpstream/AssignVariableOp26
stream_1/AssignVariableOpstream_1/AssignVariableOp26
stream_2/AssignVariableOpstream_2/AssignVariableOp26
stream_3/AssignVariableOpstream_3/AssignVariableOp26
stream_4/AssignVariableOpstream_4/AssignVariableOp26
stream_5/AssignVariableOpstream_5/AssignVariableOp26
stream_6/AssignVariableOpstream_6/AssignVariableOp26
stream_7/AssignVariableOpstream_7/AssignVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
¤
_
A__inference_dropout_layer_call_and_return_conditional_losses_3105

inputs

identity_1R
IdentityIdentityinputs*
T0*
_output_shapes
:	2

Identitya

Identity_1IdentityIdentity:output:0*
T0*
_output_shapes
:	2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs


B__inference_stream_3_layer_call_and_return_conditional_losses_4244

inputs,
(readvariableop_streaming_stream_3_states;
7conv2d_3_conv2d_readvariableop_stream_3_conv2d_3_kernel:
6conv2d_3_biasadd_readvariableop_stream_3_conv2d_3_bias
identity¢AssignVariableOp
ReadVariableOpReadVariableOp(readvariableop_streaming_stream_3_states*&
_output_shapes
:@*
dtype02
ReadVariableOp
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:@*

begin_mask*
end_mask2
strided_slice\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2strided_slice:output:0inputsconcat/axis:output:0*
N*
T0*&
_output_shapes
:@2
concat§
AssignVariableOpAssignVariableOp(readvariableop_streaming_stream_3_statesconcat:output:0^ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOpÓ
conv2d_3/Conv2D/ReadVariableOpReadVariableOp7conv2d_3_conv2d_readvariableop_stream_3_conv2d_3_kernel^AssignVariableOp*&
_output_shapes
:@@*
dtype02 
conv2d_3/Conv2D/ReadVariableOp¿
conv2d_3/Conv2DConv2Dconcat:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
conv2d_3/Conv2DÈ
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp6conv2d_3_biasadd_readvariableop_stream_3_conv2d_3_bias^AssignVariableOp*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp£
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
conv2d_3/BiasAddr
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*&
_output_shapes
:@2
conv2d_3/Relu
IdentityIdentityconv2d_3/Relu:activations:0^AssignVariableOp*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*1
_input_shapes 
:@:::2$
AssignVariableOpAssignVariableOp:N J
&
_output_shapes
:@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
¶

'__inference_conv2d_1_layer_call_fn_2643

inputs
stream_1_conv2d_1_kernel
stream_1_conv2d_1_bias
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsstream_1_conv2d_1_kernelstream_1_conv2d_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_26382
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ä

'__inference_stream_7_layer_call_fn_4353

inputs
streaming_stream_7_states
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputsstreaming_stream_7_states*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_stream_7_layer_call_and_return_conditional_losses_30782
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0**
_input_shapes
::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:
 
_user_specified_nameinputs:

_output_shapes
: 
Ì
¾
@__inference_conv2d_layer_call_and_return_conditional_losses_2605

inputs.
*conv2d_readvariableop_stream_conv2d_kernel-
)biasadd_readvariableop_stream_conv2d_bias
identityo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate¡
Conv2D/ReadVariableOpReadVariableOp*conv2d_readvariableop_stream_conv2d_kernel*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp¶
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOp)biasadd_readvariableop_stream_conv2d_bias*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2
Relu
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
¸

'__inference_conv2d_4_layer_call_fn_2742

inputs
stream_4_conv2d_4_kernel
stream_4_conv2d_4_bias
identity¢StatefulPartitionedCall¨
StatefulPartitionedCallStatefulPartitionedCallinputsstream_4_conv2d_4_kernelstream_4_conv2d_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_27372
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: "L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¤
serving_default
>
input_audio/
serving_default_input_audio:02
dense_2'
StatefulPartitionedCall:0tensorflow/serving/predict:ñ
ß¥
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer_with_weights-7

layer-9
layer-10
layer_with_weights-8
layer-11
layer_with_weights-9
layer-12
layer_with_weights-10
layer-13
	variables
trainable_variables
regularization_losses
	keras_api

signatures
+&call_and_return_all_conditional_losses
__call__
_default_save_signature"Æ 
_tf_keras_network© {"class_name": "Functional", "name": "functional_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 1, 20]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_audio"}, "name": "input_audio", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "ExpandDims", "trainable": true, "dtype": "float32", "node_def": {"name": "ExpandDims", "op": "ExpandDims", "input": ["input_1", "ExpandDims/dim"], "attr": {"Tdim": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": -1}}, "name": "tf_op_layer_ExpandDims", "inbound_nodes": [[["input_audio", 0, 0, {}]]]}, {"class_name": "Stream", "config": {"name": "stream", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "STREAM_INTERNAL_STATE_INFERENCE", "pad_time_dim": false, "cell": {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "__passive_serialization__": true}, "state_shape": [1, 3, 20, 1], "ring_buffer_size_in_time_dim": 3}, "name": "stream", "inbound_nodes": [[["tf_op_layer_ExpandDims", 0, 0, {}]]]}, {"class_name": "Stream", "config": {"name": "stream_1", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "STREAM_INTERNAL_STATE_INFERENCE", "pad_time_dim": false, "cell": {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "__passive_serialization__": true}, "state_shape": [1, 5, 18, 64], "ring_buffer_size_in_time_dim": 5}, "name": "stream_1", "inbound_nodes": [[["stream", 0, 0, {}]]]}, {"class_name": "Stream", "config": {"name": "stream_2", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "STREAM_INTERNAL_STATE_INFERENCE", "pad_time_dim": false, "cell": {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "__passive_serialization__": true}, "state_shape": [1, 5, 16, 64], "ring_buffer_size_in_time_dim": 5}, "name": "stream_2", "inbound_nodes": [[["stream_1", 0, 0, {}]]]}, {"class_name": "Stream", "config": {"name": "stream_3", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "STREAM_INTERNAL_STATE_INFERENCE", "pad_time_dim": false, "cell": {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "__passive_serialization__": true}, "state_shape": [1, 5, 14, 64], "ring_buffer_size_in_time_dim": 5}, "name": "stream_3", "inbound_nodes": [[["stream_2", 0, 0, {}]]]}, {"class_name": "Stream", "config": {"name": "stream_4", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "STREAM_INTERNAL_STATE_INFERENCE", "pad_time_dim": false, "cell": {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "__passive_serialization__": true}, "state_shape": [1, 5, 12, 64], "ring_buffer_size_in_time_dim": 5}, "name": "stream_4", "inbound_nodes": [[["stream_3", 0, 0, {}]]]}, {"class_name": "Stream", "config": {"name": "stream_5", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "STREAM_INTERNAL_STATE_INFERENCE", "pad_time_dim": false, "cell": {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "__passive_serialization__": true}, "state_shape": [1, 5, 11, 128], "ring_buffer_size_in_time_dim": 5}, "name": "stream_5", "inbound_nodes": [[["stream_4", 0, 0, {}]]]}, {"class_name": "Stream", "config": {"name": "stream_6", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "STREAM_INTERNAL_STATE_INFERENCE", "pad_time_dim": false, "cell": {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "__passive_serialization__": true}, "state_shape": [1, 3, 11, 64], "ring_buffer_size_in_time_dim": 3}, "name": "stream_6", "inbound_nodes": [[["stream_5", 0, 0, {}]]]}, {"class_name": "Stream", "config": {"name": "stream_7", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "STREAM_INTERNAL_STATE_INFERENCE", "pad_time_dim": false, "cell": {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "__passive_serialization__": true}, "state_shape": [1, 1, 11, 128], "ring_buffer_size_in_time_dim": 1}, "name": "stream_7", "inbound_nodes": [[["stream_6", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["stream_7", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 14, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}], "input_layers": [["input_audio", 0, 0]], "output_layers": [["dense_2", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 1, 20]}, "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 1, 20]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_audio"}, "name": "input_audio", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "ExpandDims", "trainable": true, "dtype": "float32", "node_def": {"name": "ExpandDims", "op": "ExpandDims", "input": ["input_1", "ExpandDims/dim"], "attr": {"Tdim": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": -1}}, "name": "tf_op_layer_ExpandDims", "inbound_nodes": [[["input_audio", 0, 0, {}]]]}, {"class_name": "Stream", "config": {"name": "stream", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "STREAM_INTERNAL_STATE_INFERENCE", "pad_time_dim": false, "cell": {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "__passive_serialization__": true}, "state_shape": [1, 3, 20, 1], "ring_buffer_size_in_time_dim": 3}, "name": "stream", "inbound_nodes": [[["tf_op_layer_ExpandDims", 0, 0, {}]]]}, {"class_name": "Stream", "config": {"name": "stream_1", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "STREAM_INTERNAL_STATE_INFERENCE", "pad_time_dim": false, "cell": {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "__passive_serialization__": true}, "state_shape": [1, 5, 18, 64], "ring_buffer_size_in_time_dim": 5}, "name": "stream_1", "inbound_nodes": [[["stream", 0, 0, {}]]]}, {"class_name": "Stream", "config": {"name": "stream_2", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "STREAM_INTERNAL_STATE_INFERENCE", "pad_time_dim": false, "cell": {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "__passive_serialization__": true}, "state_shape": [1, 5, 16, 64], "ring_buffer_size_in_time_dim": 5}, "name": "stream_2", "inbound_nodes": [[["stream_1", 0, 0, {}]]]}, {"class_name": "Stream", "config": {"name": "stream_3", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "STREAM_INTERNAL_STATE_INFERENCE", "pad_time_dim": false, "cell": {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "__passive_serialization__": true}, "state_shape": [1, 5, 14, 64], "ring_buffer_size_in_time_dim": 5}, "name": "stream_3", "inbound_nodes": [[["stream_2", 0, 0, {}]]]}, {"class_name": "Stream", "config": {"name": "stream_4", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "STREAM_INTERNAL_STATE_INFERENCE", "pad_time_dim": false, "cell": {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "__passive_serialization__": true}, "state_shape": [1, 5, 12, 64], "ring_buffer_size_in_time_dim": 5}, "name": "stream_4", "inbound_nodes": [[["stream_3", 0, 0, {}]]]}, {"class_name": "Stream", "config": {"name": "stream_5", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "STREAM_INTERNAL_STATE_INFERENCE", "pad_time_dim": false, "cell": {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "__passive_serialization__": true}, "state_shape": [1, 5, 11, 128], "ring_buffer_size_in_time_dim": 5}, "name": "stream_5", "inbound_nodes": [[["stream_4", 0, 0, {}]]]}, {"class_name": "Stream", "config": {"name": "stream_6", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "STREAM_INTERNAL_STATE_INFERENCE", "pad_time_dim": false, "cell": {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "__passive_serialization__": true}, "state_shape": [1, 3, 11, 64], "ring_buffer_size_in_time_dim": 3}, "name": "stream_6", "inbound_nodes": [[["stream_5", 0, 0, {}]]]}, {"class_name": "Stream", "config": {"name": "stream_7", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "STREAM_INTERNAL_STATE_INFERENCE", "pad_time_dim": false, "cell": {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "__passive_serialization__": true}, "state_shape": [1, 1, 11, 128], "ring_buffer_size_in_time_dim": 1}, "name": "stream_7", "inbound_nodes": [[["stream_6", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["stream_7", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 14, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}], "input_layers": [["input_audio", 0, 0]], "output_layers": [["dense_2", 0, 0]]}}}
ó"ð
_tf_keras_input_layerÐ{"class_name": "InputLayer", "name": "input_audio", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [1, 1, 20]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 1, 20]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_audio"}}
è
	variables
trainable_variables
regularization_losses
	keras_api
+&call_and_return_all_conditional_losses
__call__"×
_tf_keras_layer½{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_ExpandDims", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "ExpandDims", "trainable": true, "dtype": "float32", "node_def": {"name": "ExpandDims", "op": "ExpandDims", "input": ["input_1", "ExpandDims/dim"], "attr": {"Tdim": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": -1}}}
Ý

cell
state_shape

states
	variables
trainable_variables
regularization_losses
	keras_api
+&call_and_return_all_conditional_losses
__call__"¥	
_tf_keras_layer	{"class_name": "Stream", "name": "stream", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "stream", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "STREAM_INTERNAL_STATE_INFERENCE", "pad_time_dim": false, "cell": {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "__passive_serialization__": true}, "state_shape": [1, 3, 20, 1], "ring_buffer_size_in_time_dim": 3}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 1, 20, 1]}}
å

cell
 state_shape

!states
"	variables
#trainable_variables
$regularization_losses
%	keras_api
+&call_and_return_all_conditional_losses
__call__"­	
_tf_keras_layer	{"class_name": "Stream", "name": "stream_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "stream_1", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "STREAM_INTERNAL_STATE_INFERENCE", "pad_time_dim": false, "cell": {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "__passive_serialization__": true}, "state_shape": [1, 5, 18, 64], "ring_buffer_size_in_time_dim": 5}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 1, 18, 64]}}
å

&cell
'state_shape

(states
)	variables
*trainable_variables
+regularization_losses
,	keras_api
+&call_and_return_all_conditional_losses
__call__"­	
_tf_keras_layer	{"class_name": "Stream", "name": "stream_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "stream_2", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "STREAM_INTERNAL_STATE_INFERENCE", "pad_time_dim": false, "cell": {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "__passive_serialization__": true}, "state_shape": [1, 5, 16, 64], "ring_buffer_size_in_time_dim": 5}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 1, 16, 64]}}
å

-cell
.state_shape

/states
0	variables
1trainable_variables
2regularization_losses
3	keras_api
+&call_and_return_all_conditional_losses
__call__"­	
_tf_keras_layer	{"class_name": "Stream", "name": "stream_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "stream_3", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "STREAM_INTERNAL_STATE_INFERENCE", "pad_time_dim": false, "cell": {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "__passive_serialization__": true}, "state_shape": [1, 5, 14, 64], "ring_buffer_size_in_time_dim": 5}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 1, 14, 64]}}
æ

4cell
5state_shape

6states
7	variables
8trainable_variables
9regularization_losses
:	keras_api
+&call_and_return_all_conditional_losses
__call__"®	
_tf_keras_layer	{"class_name": "Stream", "name": "stream_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "stream_4", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "STREAM_INTERNAL_STATE_INFERENCE", "pad_time_dim": false, "cell": {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "__passive_serialization__": true}, "state_shape": [1, 5, 12, 64], "ring_buffer_size_in_time_dim": 5}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 1, 12, 64]}}
ç

;cell
<state_shape

=states
>	variables
?trainable_variables
@regularization_losses
A	keras_api
+&call_and_return_all_conditional_losses
__call__"¯	
_tf_keras_layer	{"class_name": "Stream", "name": "stream_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "stream_5", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "STREAM_INTERNAL_STATE_INFERENCE", "pad_time_dim": false, "cell": {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "__passive_serialization__": true}, "state_shape": [1, 5, 11, 128], "ring_buffer_size_in_time_dim": 5}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 1, 11, 128]}}
æ

Bcell
Cstate_shape

Dstates
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
+&call_and_return_all_conditional_losses
__call__"®	
_tf_keras_layer	{"class_name": "Stream", "name": "stream_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "stream_6", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "STREAM_INTERNAL_STATE_INFERENCE", "pad_time_dim": false, "cell": {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "__passive_serialization__": true}, "state_shape": [1, 3, 11, 64], "ring_buffer_size_in_time_dim": 3}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 1, 11, 64]}}
Ó
Icell
Jstate_shape

Kstates
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer{"class_name": "Stream", "name": "stream_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "stream_7", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "STREAM_INTERNAL_STATE_INFERENCE", "pad_time_dim": false, "cell": {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "__passive_serialization__": true}, "state_shape": [1, 1, 11, 128], "ring_buffer_size_in_time_dim": 1}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 1, 11, 128]}}
À
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
+&call_and_return_all_conditional_losses
__call__"¯
_tf_keras_layer{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}


Tkernel
Ubias
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
+&call_and_return_all_conditional_losses
__call__"à
_tf_keras_layerÆ{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1408}}}}


Zkernel
[bias
\	variables
]trainable_variables
^regularization_losses
_	keras_api
+&call_and_return_all_conditional_losses
__call__"á
_tf_keras_layerÇ{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}


`kernel
abias
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
+&call_and_return_all_conditional_losses
__call__"â
_tf_keras_layerÈ{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 14, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}}
ö
f0
g1
2
h3
i4
!5
j6
k7
(8
l9
m10
/11
n12
o13
614
p15
q16
=17
r18
s19
D20
K21
T22
U23
Z24
[25
`26
a27"
trackable_list_wrapper
¶
f0
g1
h2
i3
j4
k5
l6
m7
n8
o9
p10
q11
r12
s13
T14
U15
Z16
[17
`18
a19"
trackable_list_wrapper
 "
trackable_list_wrapper
Î
	variables
tmetrics
ulayer_metrics
vnon_trainable_variables
trainable_variables

wlayers
regularization_losses
xlayer_regularization_losses
__call__
_default_save_signature
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
ymetrics
	variables
zlayer_metrics
{non_trainable_variables
trainable_variables

|layers
regularization_losses
}layer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ñ

fkernel
gbias
~	variables
trainable_variables
regularization_losses
	keras_api
+ &call_and_return_all_conditional_losses
¡__call__"È
_tf_keras_layer®{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}}
 "
trackable_list_wrapper
/:-2streaming/stream/states
5
f0
g1
2"
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
metrics
	variables
layer_metrics
non_trainable_variables
trainable_variables
layers
regularization_losses
 layer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ø

hkernel
ibias
	variables
trainable_variables
regularization_losses
	keras_api
+¢&call_and_return_all_conditional_losses
£__call__"Í
_tf_keras_layer³{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
 "
trackable_list_wrapper
1:/@2streaming/stream_1/states
5
h0
i1
!2"
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
metrics
"	variables
layer_metrics
non_trainable_variables
#trainable_variables
layers
$regularization_losses
 layer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ø

jkernel
kbias
	variables
trainable_variables
regularization_losses
	keras_api
+¤&call_and_return_all_conditional_losses
¥__call__"Í
_tf_keras_layer³{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
 "
trackable_list_wrapper
1:/@2streaming/stream_2/states
5
j0
k1
(2"
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
metrics
)	variables
layer_metrics
non_trainable_variables
*trainable_variables
layers
+regularization_losses
 layer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ø

lkernel
mbias
	variables
trainable_variables
regularization_losses
	keras_api
+¦&call_and_return_all_conditional_losses
§__call__"Í
_tf_keras_layer³{"class_name": "Conv2D", "name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
 "
trackable_list_wrapper
1:/@2streaming/stream_3/states
5
l0
m1
/2"
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
metrics
0	variables
layer_metrics
non_trainable_variables
1trainable_variables
 layers
2regularization_losses
 ¡layer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ù

nkernel
obias
¢	variables
£trainable_variables
¤regularization_losses
¥	keras_api
+¨&call_and_return_all_conditional_losses
©__call__"Î
_tf_keras_layer´{"class_name": "Conv2D", "name": "conv2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
 "
trackable_list_wrapper
1:/@2streaming/stream_4/states
5
n0
o1
62"
trackable_list_wrapper
.
n0
o1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
¦metrics
7	variables
§layer_metrics
¨non_trainable_variables
8trainable_variables
©layers
9regularization_losses
 ªlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ù

pkernel
qbias
«	variables
¬trainable_variables
­regularization_losses
®	keras_api
+ª&call_and_return_all_conditional_losses
«__call__"Î
_tf_keras_layer´{"class_name": "Conv2D", "name": "conv2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}}
 "
trackable_list_wrapper
2:02streaming/stream_5/states
5
p0
q1
=2"
trackable_list_wrapper
.
p0
q1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
¯metrics
>	variables
°layer_metrics
±non_trainable_variables
?trainable_variables
²layers
@regularization_losses
 ³layer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ù

rkernel
sbias
´	variables
µtrainable_variables
¶regularization_losses
·	keras_api
+¬&call_and_return_all_conditional_losses
­__call__"Î
_tf_keras_layer´{"class_name": "Conv2D", "name": "conv2d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
 "
trackable_list_wrapper
1:/@2streaming/stream_6/states
5
r0
s1
D2"
trackable_list_wrapper
.
r0
s1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
¸metrics
E	variables
¹layer_metrics
ºnon_trainable_variables
Ftrainable_variables
»layers
Gregularization_losses
 ¼layer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Å
½	variables
¾trainable_variables
¿regularization_losses
À	keras_api
+®&call_and_return_all_conditional_losses
¯__call__"°
_tf_keras_layer{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
 "
trackable_list_wrapper
2:02streaming/stream_7/states
'
K0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ámetrics
L	variables
Âlayer_metrics
Ãnon_trainable_variables
Mtrainable_variables
Älayers
Nregularization_losses
 Ålayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Æmetrics
P	variables
Çlayer_metrics
Ènon_trainable_variables
Qtrainable_variables
Élayers
Rregularization_losses
 Êlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
*:(
2streaming/dense/kernel
#:!2streaming/dense/bias
.
T0
U1"
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ëmetrics
V	variables
Ìlayer_metrics
Ínon_trainable_variables
Wtrainable_variables
Îlayers
Xregularization_losses
 Ïlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
,:*
2streaming/dense_1/kernel
%:#2streaming/dense_1/bias
.
Z0
[1"
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ðmetrics
\	variables
Ñlayer_metrics
Ònon_trainable_variables
]trainable_variables
Ólayers
^regularization_losses
 Ôlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
+:)	2streaming/dense_2/kernel
$:"2streaming/dense_2/bias
.
`0
a1"
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Õmetrics
b	variables
Ölayer_metrics
×non_trainable_variables
ctrainable_variables
Ølayers
dregularization_losses
 Ùlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
.:,@2stream/conv2d/kernel
 :@2stream/conv2d/bias
2:0@@2stream_1/conv2d_1/kernel
$:"@2stream_1/conv2d_1/bias
2:0@@2stream_2/conv2d_2/kernel
$:"@2stream_2/conv2d_2/bias
2:0@@2stream_3/conv2d_3/kernel
$:"@2stream_3/conv2d_3/bias
3:1@2stream_4/conv2d_4/kernel
%:#2stream_4/conv2d_4/bias
3:1@2stream_5/conv2d_5/kernel
$:"@2stream_5/conv2d_5/bias
3:1@2stream_6/conv2d_6/kernel
%:#2stream_6/conv2d_6/bias
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
X
0
!1
(2
/3
64
=5
D6
K7"
trackable_list_wrapper

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
13"
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
f0
g1"
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
 "
trackable_list_wrapper
¶
Úmetrics
~	variables
Ûlayer_metrics
Ünon_trainable_variables
trainable_variables
Ýlayers
regularization_losses
 Þlayer_regularization_losses
¡__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ßmetrics
	variables
àlayer_metrics
ánon_trainable_variables
trainable_variables
âlayers
regularization_losses
 ãlayer_regularization_losses
£__call__
+¢&call_and_return_all_conditional_losses
'¢"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
!0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ämetrics
	variables
ålayer_metrics
ænon_trainable_variables
trainable_variables
çlayers
regularization_losses
 èlayer_regularization_losses
¥__call__
+¤&call_and_return_all_conditional_losses
'¤"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
(0"
trackable_list_wrapper
'
&0"
trackable_list_wrapper
 "
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
émetrics
	variables
êlayer_metrics
ënon_trainable_variables
trainable_variables
ìlayers
regularization_losses
 ílayer_regularization_losses
§__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
/0"
trackable_list_wrapper
'
-0"
trackable_list_wrapper
 "
trackable_list_wrapper
.
n0
o1"
trackable_list_wrapper
.
n0
o1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
îmetrics
¢	variables
ïlayer_metrics
ðnon_trainable_variables
£trainable_variables
ñlayers
¤regularization_losses
 òlayer_regularization_losses
©__call__
+¨&call_and_return_all_conditional_losses
'¨"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
60"
trackable_list_wrapper
'
40"
trackable_list_wrapper
 "
trackable_list_wrapper
.
p0
q1"
trackable_list_wrapper
.
p0
q1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ómetrics
«	variables
ôlayer_metrics
õnon_trainable_variables
¬trainable_variables
ölayers
­regularization_losses
 ÷layer_regularization_losses
«__call__
+ª&call_and_return_all_conditional_losses
'ª"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
=0"
trackable_list_wrapper
'
;0"
trackable_list_wrapper
 "
trackable_list_wrapper
.
r0
s1"
trackable_list_wrapper
.
r0
s1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ømetrics
´	variables
ùlayer_metrics
únon_trainable_variables
µtrainable_variables
ûlayers
¶regularization_losses
 ülayer_regularization_losses
­__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
D0"
trackable_list_wrapper
'
B0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ýmetrics
½	variables
þlayer_metrics
ÿnon_trainable_variables
¾trainable_variables
layers
¿regularization_losses
 layer_regularization_losses
¯__call__
+®&call_and_return_all_conditional_losses
'®"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
K0"
trackable_list_wrapper
'
I0"
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
æ2ã
F__inference_functional_1_layer_call_and_return_conditional_losses_3926
F__inference_functional_1_layer_call_and_return_conditional_losses_3712
F__inference_functional_1_layer_call_and_return_conditional_losses_4067
F__inference_functional_1_layer_call_and_return_conditional_losses_3571À
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
kwonlydefaultsª 
annotationsª *
 
ú2÷
+__inference_functional_1_layer_call_fn_4133
+__inference_functional_1_layer_call_fn_3745
+__inference_functional_1_layer_call_fn_4100
+__inference_functional_1_layer_call_fn_3778À
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
kwonlydefaultsª 
annotationsª *
 
å2â
__inference__wrapped_model_2577¾
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
annotationsª *.¢+
)&
input_audioÿÿÿÿÿÿÿÿÿ
ú2÷
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_4139¢
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
annotationsª *
 
ß2Ü
5__inference_tf_op_layer_ExpandDims_layer_call_fn_4144¢
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
annotationsª *
 
ê2ç
@__inference_stream_layer_call_and_return_conditional_losses_4163¢
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
annotationsª *
 
Ï2Ì
%__inference_stream_layer_call_fn_4171¢
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
annotationsª *
 
ì2é
B__inference_stream_1_layer_call_and_return_conditional_losses_4190¢
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
annotationsª *
 
Ñ2Î
'__inference_stream_1_layer_call_fn_4198¢
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
annotationsª *
 
ì2é
B__inference_stream_2_layer_call_and_return_conditional_losses_4217¢
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
annotationsª *
 
Ñ2Î
'__inference_stream_2_layer_call_fn_4225¢
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
annotationsª *
 
ì2é
B__inference_stream_3_layer_call_and_return_conditional_losses_4244¢
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
annotationsª *
 
Ñ2Î
'__inference_stream_3_layer_call_fn_4252¢
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
annotationsª *
 
ì2é
B__inference_stream_4_layer_call_and_return_conditional_losses_4271¢
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
annotationsª *
 
Ñ2Î
'__inference_stream_4_layer_call_fn_4279¢
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
annotationsª *
 
ì2é
B__inference_stream_5_layer_call_and_return_conditional_losses_4298¢
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
annotationsª *
 
Ñ2Î
'__inference_stream_5_layer_call_fn_4306¢
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
annotationsª *
 
ì2é
B__inference_stream_6_layer_call_and_return_conditional_losses_4325¢
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
annotationsª *
 
Ñ2Î
'__inference_stream_6_layer_call_fn_4333¢
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
annotationsª *
 
ì2é
B__inference_stream_7_layer_call_and_return_conditional_losses_4347¢
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
annotationsª *
 
Ñ2Î
'__inference_stream_7_layer_call_fn_4353¢
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
annotationsª *
 
À2½
A__inference_dropout_layer_call_and_return_conditional_losses_4370
A__inference_dropout_layer_call_and_return_conditional_losses_4365´
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
kwonlydefaultsª 
annotationsª *
 
2
&__inference_dropout_layer_call_fn_4375
&__inference_dropout_layer_call_fn_4380´
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
kwonlydefaultsª 
annotationsª *
 
é2æ
?__inference_dense_layer_call_and_return_conditional_losses_4390¢
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
annotationsª *
 
Î2Ë
$__inference_dense_layer_call_fn_4397¢
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
annotationsª *
 
ë2è
A__inference_dense_1_layer_call_and_return_conditional_losses_4408¢
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
annotationsª *
 
Ð2Í
&__inference_dense_1_layer_call_fn_4415¢
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
annotationsª *
 
ë2è
A__inference_dense_2_layer_call_and_return_conditional_losses_4425¢
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
annotationsª *
 
Ð2Í
&__inference_dense_2_layer_call_fn_4432¢
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
annotationsª *
 
5B3
"__inference_signature_wrapper_3423input_audio
2
@__inference_conv2d_layer_call_and_return_conditional_losses_2590×
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
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
%__inference_conv2d_layer_call_fn_2610×
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
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¡2
B__inference_conv2d_1_layer_call_and_return_conditional_losses_2623×
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
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
2
'__inference_conv2d_1_layer_call_fn_2643×
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
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
¡2
B__inference_conv2d_2_layer_call_and_return_conditional_losses_2656×
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
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
2
'__inference_conv2d_2_layer_call_fn_2676×
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
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
¡2
B__inference_conv2d_3_layer_call_and_return_conditional_losses_2689×
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
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
2
'__inference_conv2d_3_layer_call_fn_2709×
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
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
¡2
B__inference_conv2d_4_layer_call_and_return_conditional_losses_2722×
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
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
2
'__inference_conv2d_4_layer_call_fn_2742×
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
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
¢2
B__inference_conv2d_5_layer_call_and_return_conditional_losses_2755Ø
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
annotationsª *8¢5
30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
'__inference_conv2d_5_layer_call_fn_2775Ø
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
annotationsª *8¢5
30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¡2
B__inference_conv2d_6_layer_call_and_return_conditional_losses_2788×
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
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
2
'__inference_conv2d_6_layer_call_fn_2808×
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
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
¨2¥¢
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
annotationsª *
 
¨2¥¢
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
annotationsª *
 ¦
__inference__wrapped_model_2577fg!hi(jk/lm6no=pqDrsKTUZ[`a8¢5
.¢+
)&
input_audioÿÿÿÿÿÿÿÿÿ
ª "(ª%
#
dense_2
dense_2×
B__inference_conv2d_1_layer_call_and_return_conditional_losses_2623hiI¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 ¯
'__inference_conv2d_1_layer_call_fn_2643hiI¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@×
B__inference_conv2d_2_layer_call_and_return_conditional_losses_2656jkI¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 ¯
'__inference_conv2d_2_layer_call_fn_2676jkI¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@×
B__inference_conv2d_3_layer_call_and_return_conditional_losses_2689lmI¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 ¯
'__inference_conv2d_3_layer_call_fn_2709lmI¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ø
B__inference_conv2d_4_layer_call_and_return_conditional_losses_2722noI¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 °
'__inference_conv2d_4_layer_call_fn_2742noI¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØ
B__inference_conv2d_5_layer_call_and_return_conditional_losses_2755pqJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 °
'__inference_conv2d_5_layer_call_fn_2775pqJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ø
B__inference_conv2d_6_layer_call_and_return_conditional_losses_2788rsI¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 °
'__inference_conv2d_6_layer_call_fn_2808rsI¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÕ
@__inference_conv2d_layer_call_and_return_conditional_losses_2590fgI¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 ­
%__inference_conv2d_layer_call_fn_2610fgI¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
A__inference_dense_1_layer_call_and_return_conditional_losses_4408LZ['¢$
¢

inputs	
ª "¢

0	
 i
&__inference_dense_1_layer_call_fn_4415?Z['¢$
¢

inputs	
ª "	
A__inference_dense_2_layer_call_and_return_conditional_losses_4425K`a'¢$
¢

inputs	
ª "¢

0
 h
&__inference_dense_2_layer_call_fn_4432>`a'¢$
¢

inputs	
ª "
?__inference_dense_layer_call_and_return_conditional_losses_4390LTU'¢$
¢

inputs	
ª "¢

0	
 g
$__inference_dense_layer_call_fn_4397?TU'¢$
¢

inputs	
ª "	
A__inference_dropout_layer_call_and_return_conditional_losses_4365L+¢(
!¢

inputs	
p
ª "¢

0	
 
A__inference_dropout_layer_call_and_return_conditional_losses_4370L+¢(
!¢

inputs	
p 
ª "¢

0	
 i
&__inference_dropout_layer_call_fn_4375?+¢(
!¢

inputs	
p
ª "	i
&__inference_dropout_layer_call_fn_4380?+¢(
!¢

inputs	
p 
ª "	Ã
F__inference_functional_1_layer_call_and_return_conditional_losses_3571yfg!hi(jk/lm6no=pqDrsKTUZ[`a;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "¢

0
 Ã
F__inference_functional_1_layer_call_and_return_conditional_losses_3712yfg!hi(jk/lm6no=pqDrsKTUZ[`a;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "¢

0
 È
F__inference_functional_1_layer_call_and_return_conditional_losses_3926~fg!hi(jk/lm6no=pqDrsKTUZ[`a@¢=
6¢3
)&
input_audioÿÿÿÿÿÿÿÿÿ
p

 
ª "¢

0
 È
F__inference_functional_1_layer_call_and_return_conditional_losses_4067~fg!hi(jk/lm6no=pqDrsKTUZ[`a@¢=
6¢3
)&
input_audioÿÿÿÿÿÿÿÿÿ
p 

 
ª "¢

0
 
+__inference_functional_1_layer_call_fn_3745lfg!hi(jk/lm6no=pqDrsKTUZ[`a;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "
+__inference_functional_1_layer_call_fn_3778lfg!hi(jk/lm6no=pqDrsKTUZ[`a;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª " 
+__inference_functional_1_layer_call_fn_4100qfg!hi(jk/lm6no=pqDrsKTUZ[`a@¢=
6¢3
)&
input_audioÿÿÿÿÿÿÿÿÿ
p

 
ª " 
+__inference_functional_1_layer_call_fn_4133qfg!hi(jk/lm6no=pqDrsKTUZ[`a@¢=
6¢3
)&
input_audioÿÿÿÿÿÿÿÿÿ
p 

 
ª "¯
"__inference_signature_wrapper_3423fg!hi(jk/lm6no=pqDrsKTUZ[`a>¢;
¢ 
4ª1
/
input_audio 
input_audio"(ª%
#
dense_2
dense_2¡
B__inference_stream_1_layer_call_and_return_conditional_losses_4190[!hi.¢+
$¢!

inputs@
ª "$¢!

0@
 y
'__inference_stream_1_layer_call_fn_4198N!hi.¢+
$¢!

inputs@
ª "@¡
B__inference_stream_2_layer_call_and_return_conditional_losses_4217[(jk.¢+
$¢!

inputs@
ª "$¢!

0@
 y
'__inference_stream_2_layer_call_fn_4225N(jk.¢+
$¢!

inputs@
ª "@¡
B__inference_stream_3_layer_call_and_return_conditional_losses_4244[/lm.¢+
$¢!

inputs@
ª "$¢!

0@
 y
'__inference_stream_3_layer_call_fn_4252N/lm.¢+
$¢!

inputs@
ª "@¢
B__inference_stream_4_layer_call_and_return_conditional_losses_4271\6no.¢+
$¢!

inputs@
ª "%¢"

0
 z
'__inference_stream_4_layer_call_fn_4279O6no.¢+
$¢!

inputs@
ª "¢
B__inference_stream_5_layer_call_and_return_conditional_losses_4298\=pq/¢,
%¢"
 
inputs
ª "$¢!

0@
 z
'__inference_stream_5_layer_call_fn_4306O=pq/¢,
%¢"
 
inputs
ª "@¢
B__inference_stream_6_layer_call_and_return_conditional_losses_4325\Drs.¢+
$¢!

inputs@
ª "%¢"

0
 z
'__inference_stream_6_layer_call_fn_4333ODrs.¢+
$¢!

inputs@
ª "
B__inference_stream_7_layer_call_and_return_conditional_losses_4347SK/¢,
%¢"
 
inputs
ª "¢

0	
 q
'__inference_stream_7_layer_call_fn_4353FK/¢,
%¢"
 
inputs
ª "	
@__inference_stream_layer_call_and_return_conditional_losses_4163[fg.¢+
$¢!

inputs
ª "$¢!

0@
 w
%__inference_stream_layer_call_fn_4171Nfg.¢+
$¢!

inputs
ª "@¦
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_4139R*¢'
 ¢

inputs
ª "$¢!

0
 ~
5__inference_tf_op_layer_ExpandDims_layer_call_fn_4144E*¢'
 ¢

inputs
ª "