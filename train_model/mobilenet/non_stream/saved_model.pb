ёс(
╤г
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
dtypetypeИ
╛
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
executor_typestring И
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.3.02v2.3.0-rc2-23-gb36436b0878■щ 
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
К
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namebatch_normalization/gamma
Г
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
: *
dtype0
И
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namebatch_normalization/beta
Б
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
: *
dtype0
Ц
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!batch_normalization/moving_mean
П
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
: *
dtype0
Ю
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#batch_normalization/moving_variance
Ч
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
: *
dtype0
ж
!depthwise_conv2d/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!depthwise_conv2d/depthwise_kernel
Я
5depthwise_conv2d/depthwise_kernel/Read/ReadVariableOpReadVariableOp!depthwise_conv2d/depthwise_kernel*&
_output_shapes
: *
dtype0
О
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_1/gamma
З
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
: *
dtype0
М
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_1/beta
Е
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
: *
dtype0
Ъ
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_1/moving_mean
У
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
: *
dtype0
в
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_1/moving_variance
Ы
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
: *
dtype0
В
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:  *
dtype0
О
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_2/gamma
З
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
: *
dtype0
М
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_2/beta
Е
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
: *
dtype0
Ъ
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_2/moving_mean
У
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
: *
dtype0
в
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_2/moving_variance
Ы
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
: *
dtype0
к
#depthwise_conv2d_1/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#depthwise_conv2d_1/depthwise_kernel
г
7depthwise_conv2d_1/depthwise_kernel/Read/ReadVariableOpReadVariableOp#depthwise_conv2d_1/depthwise_kernel*&
_output_shapes
: *
dtype0
О
batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_3/gamma
З
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes
: *
dtype0
М
batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_3/beta
Е
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes
: *
dtype0
Ъ
!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_3/moving_mean
У
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes
: *
dtype0
в
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_3/moving_variance
Ы
9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes
: *
dtype0
В
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
: @*
dtype0
О
batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_4/gamma
З
/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
_output_shapes
:@*
dtype0
М
batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_4/beta
Е
.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
_output_shapes
:@*
dtype0
Ъ
!batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_4/moving_mean
У
5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
_output_shapes
:@*
dtype0
в
%batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_4/moving_variance
Ы
9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
_output_shapes
:@*
dtype0
к
#depthwise_conv2d_2/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#depthwise_conv2d_2/depthwise_kernel
г
7depthwise_conv2d_2/depthwise_kernel/Read/ReadVariableOpReadVariableOp#depthwise_conv2d_2/depthwise_kernel*&
_output_shapes
:@*
dtype0
О
batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_5/gamma
З
/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_5/gamma*
_output_shapes
:@*
dtype0
М
batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_5/beta
Е
.batch_normalization_5/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_5/beta*
_output_shapes
:@*
dtype0
Ъ
!batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_5/moving_mean
У
5batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_5/moving_mean*
_output_shapes
:@*
dtype0
в
%batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_5/moving_variance
Ы
9batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_5/moving_variance*
_output_shapes
:@*
dtype0
Г
conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А* 
shared_nameconv2d_3/kernel
|
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*'
_output_shapes
:@А*
dtype0
П
batch_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*,
shared_namebatch_normalization_6/gamma
И
/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_6/gamma*
_output_shapes	
:А*
dtype0
Н
batch_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*+
shared_namebatch_normalization_6/beta
Ж
.batch_normalization_6/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_6/beta*
_output_shapes	
:А*
dtype0
Ы
!batch_normalization_6/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*2
shared_name#!batch_normalization_6/moving_mean
Ф
5batch_normalization_6/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_6/moving_mean*
_output_shapes	
:А*
dtype0
г
%batch_normalization_6/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%batch_normalization_6/moving_variance
Ь
9batch_normalization_6/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_6/moving_variance*
_output_shapes	
:А*
dtype0
л
#depthwise_conv2d_3/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*4
shared_name%#depthwise_conv2d_3/depthwise_kernel
д
7depthwise_conv2d_3/depthwise_kernel/Read/ReadVariableOpReadVariableOp#depthwise_conv2d_3/depthwise_kernel*'
_output_shapes
:А*
dtype0
П
batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*,
shared_namebatch_normalization_7/gamma
И
/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_7/gamma*
_output_shapes	
:А*
dtype0
Н
batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*+
shared_namebatch_normalization_7/beta
Ж
.batch_normalization_7/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_7/beta*
_output_shapes	
:А*
dtype0
Ы
!batch_normalization_7/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*2
shared_name#!batch_normalization_7/moving_mean
Ф
5batch_normalization_7/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_7/moving_mean*
_output_shapes	
:А*
dtype0
г
%batch_normalization_7/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%batch_normalization_7/moving_variance
Ь
9batch_normalization_7/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_7/moving_variance*
_output_shapes	
:А*
dtype0
Д
conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА* 
shared_nameconv2d_4/kernel
}
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*(
_output_shapes
:АА*
dtype0
П
batch_normalization_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*,
shared_namebatch_normalization_8/gamma
И
/batch_normalization_8/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_8/gamma*
_output_shapes	
:А*
dtype0
Н
batch_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*+
shared_namebatch_normalization_8/beta
Ж
.batch_normalization_8/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_8/beta*
_output_shapes	
:А*
dtype0
Ы
!batch_normalization_8/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*2
shared_name#!batch_normalization_8/moving_mean
Ф
5batch_normalization_8/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_8/moving_mean*
_output_shapes	
:А*
dtype0
г
%batch_normalization_8/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%batch_normalization_8/moving_variance
Ь
9batch_normalization_8/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_8/moving_variance*
_output_shapes	
:А*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	А*
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

NoOpNoOp
╓Л
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*РЛ
valueЕЛBБЛ B∙К
П
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
layer-22
layer_with_weights-14
layer-23
layer_with_weights-15
layer-24
layer-25
layer_with_weights-16
layer-26
layer_with_weights-17
layer-27
layer-28
layer-29
layer-30
 layer_with_weights-18
 layer-31
!trainable_variables
"	variables
#regularization_losses
$	keras_api
%
signatures
 
R
&trainable_variables
'	variables
(regularization_losses
)	keras_api
^

*kernel
+trainable_variables
,	variables
-regularization_losses
.	keras_api
Ч
/axis
	0gamma
1beta
2moving_mean
3moving_variance
4trainable_variables
5	variables
6regularization_losses
7	keras_api
R
8trainable_variables
9	variables
:regularization_losses
;	keras_api
h
<depthwise_kernel
=trainable_variables
>	variables
?regularization_losses
@	keras_api
Ч
Aaxis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance
Ftrainable_variables
G	variables
Hregularization_losses
I	keras_api
R
Jtrainable_variables
K	variables
Lregularization_losses
M	keras_api
^

Nkernel
Otrainable_variables
P	variables
Qregularization_losses
R	keras_api
Ч
Saxis
	Tgamma
Ubeta
Vmoving_mean
Wmoving_variance
Xtrainable_variables
Y	variables
Zregularization_losses
[	keras_api
R
\trainable_variables
]	variables
^regularization_losses
_	keras_api
h
`depthwise_kernel
atrainable_variables
b	variables
cregularization_losses
d	keras_api
Ч
eaxis
	fgamma
gbeta
hmoving_mean
imoving_variance
jtrainable_variables
k	variables
lregularization_losses
m	keras_api
R
ntrainable_variables
o	variables
pregularization_losses
q	keras_api
^

rkernel
strainable_variables
t	variables
uregularization_losses
v	keras_api
Ч
waxis
	xgamma
ybeta
zmoving_mean
{moving_variance
|trainable_variables
}	variables
~regularization_losses
	keras_api
V
Аtrainable_variables
Б	variables
Вregularization_losses
Г	keras_api
m
Дdepthwise_kernel
Еtrainable_variables
Ж	variables
Зregularization_losses
И	keras_api
а
	Йaxis

Кgamma
	Лbeta
Мmoving_mean
Нmoving_variance
Оtrainable_variables
П	variables
Рregularization_losses
С	keras_api
V
Тtrainable_variables
У	variables
Фregularization_losses
Х	keras_api
c
Цkernel
Чtrainable_variables
Ш	variables
Щregularization_losses
Ъ	keras_api
а
	Ыaxis

Ьgamma
	Эbeta
Юmoving_mean
Яmoving_variance
аtrainable_variables
б	variables
вregularization_losses
г	keras_api
V
дtrainable_variables
е	variables
жregularization_losses
з	keras_api
m
иdepthwise_kernel
йtrainable_variables
к	variables
лregularization_losses
м	keras_api
а
	нaxis

оgamma
	пbeta
░moving_mean
▒moving_variance
▓trainable_variables
│	variables
┤regularization_losses
╡	keras_api
V
╢trainable_variables
╖	variables
╕regularization_losses
╣	keras_api
c
║kernel
╗trainable_variables
╝	variables
╜regularization_losses
╛	keras_api
а
	┐axis

└gamma
	┴beta
┬moving_mean
├moving_variance
─trainable_variables
┼	variables
╞regularization_losses
╟	keras_api
V
╚trainable_variables
╔	variables
╩regularization_losses
╦	keras_api
V
╠trainable_variables
═	variables
╬regularization_losses
╧	keras_api
V
╨trainable_variables
╤	variables
╥regularization_losses
╙	keras_api
n
╘kernel
	╒bias
╓trainable_variables
╫	variables
╪regularization_losses
┘	keras_api
ь
*0
01
12
<3
B4
C5
N6
T7
U8
`9
f10
g11
r12
x13
y14
Д15
К16
Л17
Ц18
Ь19
Э20
и21
о22
п23
║24
└25
┴26
╘27
╒28
Д
*0
01
12
23
34
<5
B6
C7
D8
E9
N10
T11
U12
V13
W14
`15
f16
g17
h18
i19
r20
x21
y22
z23
{24
Д25
К26
Л27
М28
Н29
Ц30
Ь31
Э32
Ю33
Я34
и35
о36
п37
░38
▒39
║40
└41
┴42
┬43
├44
╘45
╒46
 
▓
┌metrics
█layers
!trainable_variables
▄layer_metrics
▌non_trainable_variables
 ▐layer_regularization_losses
"	variables
#regularization_losses
 
 
 
 
▓
▀metrics
рlayers
&trainable_variables
сlayer_metrics
тnon_trainable_variables
 уlayer_regularization_losses
'	variables
(regularization_losses
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE

*0

*0
 
▓
фmetrics
хlayers
+trainable_variables
цlayer_metrics
чnon_trainable_variables
 шlayer_regularization_losses
,	variables
-regularization_losses
 
db
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

00
11

00
11
22
33
 
▓
щmetrics
ъlayers
4trainable_variables
ыlayer_metrics
ьnon_trainable_variables
 эlayer_regularization_losses
5	variables
6regularization_losses
 
 
 
▓
юmetrics
яlayers
8trainable_variables
Ёlayer_metrics
ёnon_trainable_variables
 Єlayer_regularization_losses
9	variables
:regularization_losses
wu
VARIABLE_VALUE!depthwise_conv2d/depthwise_kernel@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE

<0

<0
 
▓
єmetrics
Їlayers
=trainable_variables
їlayer_metrics
Ўnon_trainable_variables
 ўlayer_regularization_losses
>	variables
?regularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

B0
C1

B0
C1
D2
E3
 
▓
°metrics
∙layers
Ftrainable_variables
·layer_metrics
√non_trainable_variables
 №layer_regularization_losses
G	variables
Hregularization_losses
 
 
 
▓
¤metrics
■layers
Jtrainable_variables
 layer_metrics
Аnon_trainable_variables
 Бlayer_regularization_losses
K	variables
Lregularization_losses
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE

N0

N0
 
▓
Вmetrics
Гlayers
Otrainable_variables
Дlayer_metrics
Еnon_trainable_variables
 Жlayer_regularization_losses
P	variables
Qregularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_2/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_2/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_2/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_2/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

T0
U1

T0
U1
V2
W3
 
▓
Зmetrics
Иlayers
Xtrainable_variables
Йlayer_metrics
Кnon_trainable_variables
 Лlayer_regularization_losses
Y	variables
Zregularization_losses
 
 
 
▓
Мmetrics
Нlayers
\trainable_variables
Оlayer_metrics
Пnon_trainable_variables
 Рlayer_regularization_losses
]	variables
^regularization_losses
yw
VARIABLE_VALUE#depthwise_conv2d_1/depthwise_kernel@layer_with_weights-6/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE

`0

`0
 
▓
Сmetrics
Тlayers
atrainable_variables
Уlayer_metrics
Фnon_trainable_variables
 Хlayer_regularization_losses
b	variables
cregularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_3/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_3/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_3/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_3/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

f0
g1

f0
g1
h2
i3
 
▓
Цmetrics
Чlayers
jtrainable_variables
Шlayer_metrics
Щnon_trainable_variables
 Ъlayer_regularization_losses
k	variables
lregularization_losses
 
 
 
▓
Ыmetrics
Ьlayers
ntrainable_variables
Эlayer_metrics
Юnon_trainable_variables
 Яlayer_regularization_losses
o	variables
pregularization_losses
[Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE

r0

r0
 
▓
аmetrics
бlayers
strainable_variables
вlayer_metrics
гnon_trainable_variables
 дlayer_regularization_losses
t	variables
uregularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_4/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_4/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_4/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_4/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

x0
y1

x0
y1
z2
{3
 
▓
еmetrics
жlayers
|trainable_variables
зlayer_metrics
иnon_trainable_variables
 йlayer_regularization_losses
}	variables
~regularization_losses
 
 
 
╡
кmetrics
лlayers
Аtrainable_variables
мlayer_metrics
нnon_trainable_variables
 оlayer_regularization_losses
Б	variables
Вregularization_losses
zx
VARIABLE_VALUE#depthwise_conv2d_2/depthwise_kernelAlayer_with_weights-10/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE

Д0

Д0
 
╡
пmetrics
░layers
Еtrainable_variables
▒layer_metrics
▓non_trainable_variables
 │layer_regularization_losses
Ж	variables
Зregularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_5/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_5/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_5/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_5/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

К0
Л1
 
К0
Л1
М2
Н3
 
╡
┤metrics
╡layers
Оtrainable_variables
╢layer_metrics
╖non_trainable_variables
 ╕layer_regularization_losses
П	variables
Рregularization_losses
 
 
 
╡
╣metrics
║layers
Тtrainable_variables
╗layer_metrics
╝non_trainable_variables
 ╜layer_regularization_losses
У	variables
Фregularization_losses
\Z
VARIABLE_VALUEconv2d_3/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE

Ц0

Ц0
 
╡
╛metrics
┐layers
Чtrainable_variables
└layer_metrics
┴non_trainable_variables
 ┬layer_regularization_losses
Ш	variables
Щregularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_6/gamma6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_6/beta5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_6/moving_mean<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_6/moving_variance@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

Ь0
Э1
 
Ь0
Э1
Ю2
Я3
 
╡
├metrics
─layers
аtrainable_variables
┼layer_metrics
╞non_trainable_variables
 ╟layer_regularization_losses
б	variables
вregularization_losses
 
 
 
╡
╚metrics
╔layers
дtrainable_variables
╩layer_metrics
╦non_trainable_variables
 ╠layer_regularization_losses
е	variables
жregularization_losses
zx
VARIABLE_VALUE#depthwise_conv2d_3/depthwise_kernelAlayer_with_weights-14/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE

и0

и0
 
╡
═metrics
╬layers
йtrainable_variables
╧layer_metrics
╨non_trainable_variables
 ╤layer_regularization_losses
к	variables
лregularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_7/gamma6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_7/beta5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_7/moving_mean<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_7/moving_variance@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

о0
п1
 
о0
п1
░2
▒3
 
╡
╥metrics
╙layers
▓trainable_variables
╘layer_metrics
╒non_trainable_variables
 ╓layer_regularization_losses
│	variables
┤regularization_losses
 
 
 
╡
╫metrics
╪layers
╢trainable_variables
┘layer_metrics
┌non_trainable_variables
 █layer_regularization_losses
╖	variables
╕regularization_losses
\Z
VARIABLE_VALUEconv2d_4/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE

║0

║0
 
╡
▄metrics
▌layers
╗trainable_variables
▐layer_metrics
▀non_trainable_variables
 рlayer_regularization_losses
╝	variables
╜regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_8/gamma6layer_with_weights-17/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_8/beta5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_8/moving_mean<layer_with_weights-17/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_8/moving_variance@layer_with_weights-17/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

└0
┴1
 
└0
┴1
┬2
├3
 
╡
сmetrics
тlayers
─trainable_variables
уlayer_metrics
фnon_trainable_variables
 хlayer_regularization_losses
┼	variables
╞regularization_losses
 
 
 
╡
цmetrics
чlayers
╚trainable_variables
шlayer_metrics
щnon_trainable_variables
 ъlayer_regularization_losses
╔	variables
╩regularization_losses
 
 
 
╡
ыmetrics
ьlayers
╠trainable_variables
эlayer_metrics
юnon_trainable_variables
 яlayer_regularization_losses
═	variables
╬regularization_losses
 
 
 
╡
Ёmetrics
ёlayers
╨trainable_variables
Єlayer_metrics
єnon_trainable_variables
 Їlayer_regularization_losses
╤	variables
╥regularization_losses
YW
VARIABLE_VALUEdense/kernel7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUE
dense/bias5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUE

╘0
╒1

╘0
╒1
 
╡
їmetrics
Ўlayers
╓trainable_variables
ўlayer_metrics
°non_trainable_variables
 ∙layer_regularization_losses
╫	variables
╪regularization_losses
 
Ў
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
 
О
20
31
D2
E3
V4
W5
h6
i7
z8
{9
М10
Н11
Ю12
Я13
░14
▒15
┬16
├17
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
20
31
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
D0
E1
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
V0
W1
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
h0
i1
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
z0
{1
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
М0
Н1
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
Ю0
Я1
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
░0
▒1
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
┬0
├1
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
Ч
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d/kernelbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variance!depthwise_conv2d/depthwise_kernelbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_1/kernelbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variance#depthwise_conv2d_1/depthwise_kernelbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv2d_2/kernelbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variance#depthwise_conv2d_2/depthwise_kernelbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_varianceconv2d_3/kernelbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_variance#depthwise_conv2d_3/depthwise_kernelbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_varianceconv2d_4/kernelbatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_variancedense/kernel
dense/bias*;
Tin4
220*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*Q
_read_only_resource_inputs3
1/	
 !"#$%&'()*+,-./*-
config_proto

CPU

GPU 2J 8В *+
f&R$
"__inference_signature_wrapper_2913
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ж
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp5depthwise_conv2d/depthwise_kernel/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp7depthwise_conv2d_1/depthwise_kernel/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp/batch_normalization_4/gamma/Read/ReadVariableOp.batch_normalization_4/beta/Read/ReadVariableOp5batch_normalization_4/moving_mean/Read/ReadVariableOp9batch_normalization_4/moving_variance/Read/ReadVariableOp7depthwise_conv2d_2/depthwise_kernel/Read/ReadVariableOp/batch_normalization_5/gamma/Read/ReadVariableOp.batch_normalization_5/beta/Read/ReadVariableOp5batch_normalization_5/moving_mean/Read/ReadVariableOp9batch_normalization_5/moving_variance/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp/batch_normalization_6/gamma/Read/ReadVariableOp.batch_normalization_6/beta/Read/ReadVariableOp5batch_normalization_6/moving_mean/Read/ReadVariableOp9batch_normalization_6/moving_variance/Read/ReadVariableOp7depthwise_conv2d_3/depthwise_kernel/Read/ReadVariableOp/batch_normalization_7/gamma/Read/ReadVariableOp.batch_normalization_7/beta/Read/ReadVariableOp5batch_normalization_7/moving_mean/Read/ReadVariableOp9batch_normalization_7/moving_variance/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp/batch_normalization_8/gamma/Read/ReadVariableOp.batch_normalization_8/beta/Read/ReadVariableOp5batch_normalization_8/moving_mean/Read/ReadVariableOp9batch_normalization_8/moving_variance/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpConst*<
Tin5
321*
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
GPU 2J 8В *&
f!R
__inference__traced_save_5221
╒
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variance!depthwise_conv2d/depthwise_kernelbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_1/kernelbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variance#depthwise_conv2d_1/depthwise_kernelbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv2d_2/kernelbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variance#depthwise_conv2d_2/depthwise_kernelbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_varianceconv2d_3/kernelbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_variance#depthwise_conv2d_3/depthwise_kernelbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_varianceconv2d_4/kernelbatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_variancedense/kernel
dense/bias*;
Tin4
220*
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
GPU 2J 8В *)
f$R"
 __inference__traced_restore_5372ф┐
у
О
.__inference_depthwise_conv2d_layer_call_fn_832

inputs%
!depthwise_conv2d_depthwise_kernel
identityИвStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinputs!depthwise_conv2d_depthwise_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_depthwise_conv2d_layer_call_and_return_conditional_losses_8282
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+                            :22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
п
]
A__inference_re_lu_5_layer_call_and_return_conditional_losses_2183

inputs
identityP
Relu6Relu6inputs*
T0*&
_output_shapes
:@2
Relu6f
IdentityIdentityRelu6:activations:0*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*%
_input_shapes
:@:N J
&
_output_shapes
:@
 
_user_specified_nameinputs
║├
╡
F__inference_functional_1_layer_call_and_return_conditional_losses_3771

inputs.
*conv2d_conv2d_readvariableop_conv2d_kernel@
<batch_normalization_readvariableop_batch_normalization_gammaA
=batch_normalization_readvariableop_1_batch_normalization_betaW
Sbatch_normalization_fusedbatchnormv3_readvariableop_batch_normalization_moving_mean]
Ybatch_normalization_fusedbatchnormv3_readvariableop_1_batch_normalization_moving_varianceO
Kdepthwise_conv2d_depthwise_readvariableop_depthwise_conv2d_depthwise_kernelD
@batch_normalization_1_readvariableop_batch_normalization_1_gammaE
Abatch_normalization_1_readvariableop_1_batch_normalization_1_beta[
Wbatch_normalization_1_fusedbatchnormv3_readvariableop_batch_normalization_1_moving_meana
]batch_normalization_1_fusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance2
.conv2d_1_conv2d_readvariableop_conv2d_1_kernelD
@batch_normalization_2_readvariableop_batch_normalization_2_gammaE
Abatch_normalization_2_readvariableop_1_batch_normalization_2_beta[
Wbatch_normalization_2_fusedbatchnormv3_readvariableop_batch_normalization_2_moving_meana
]batch_normalization_2_fusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_varianceS
Odepthwise_conv2d_1_depthwise_readvariableop_depthwise_conv2d_1_depthwise_kernelD
@batch_normalization_3_readvariableop_batch_normalization_3_gammaE
Abatch_normalization_3_readvariableop_1_batch_normalization_3_beta[
Wbatch_normalization_3_fusedbatchnormv3_readvariableop_batch_normalization_3_moving_meana
]batch_normalization_3_fusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance2
.conv2d_2_conv2d_readvariableop_conv2d_2_kernelD
@batch_normalization_4_readvariableop_batch_normalization_4_gammaE
Abatch_normalization_4_readvariableop_1_batch_normalization_4_beta[
Wbatch_normalization_4_fusedbatchnormv3_readvariableop_batch_normalization_4_moving_meana
]batch_normalization_4_fusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_varianceS
Odepthwise_conv2d_2_depthwise_readvariableop_depthwise_conv2d_2_depthwise_kernelD
@batch_normalization_5_readvariableop_batch_normalization_5_gammaE
Abatch_normalization_5_readvariableop_1_batch_normalization_5_beta[
Wbatch_normalization_5_fusedbatchnormv3_readvariableop_batch_normalization_5_moving_meana
]batch_normalization_5_fusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance2
.conv2d_3_conv2d_readvariableop_conv2d_3_kernelD
@batch_normalization_6_readvariableop_batch_normalization_6_gammaE
Abatch_normalization_6_readvariableop_1_batch_normalization_6_beta[
Wbatch_normalization_6_fusedbatchnormv3_readvariableop_batch_normalization_6_moving_meana
]batch_normalization_6_fusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_varianceS
Odepthwise_conv2d_3_depthwise_readvariableop_depthwise_conv2d_3_depthwise_kernelD
@batch_normalization_7_readvariableop_batch_normalization_7_gammaE
Abatch_normalization_7_readvariableop_1_batch_normalization_7_beta[
Wbatch_normalization_7_fusedbatchnormv3_readvariableop_batch_normalization_7_moving_meana
]batch_normalization_7_fusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance2
.conv2d_4_conv2d_readvariableop_conv2d_4_kernelD
@batch_normalization_8_readvariableop_batch_normalization_8_gammaE
Abatch_normalization_8_readvariableop_1_batch_normalization_8_beta[
Wbatch_normalization_8_fusedbatchnormv3_readvariableop_batch_normalization_8_moving_meana
]batch_normalization_8_fusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance,
(dense_matmul_readvariableop_dense_kernel+
'dense_biasadd_readvariableop_dense_bias
identityИР
%tf_op_layer_ExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%tf_op_layer_ExpandDims/ExpandDims/dim╠
!tf_op_layer_ExpandDims/ExpandDims
ExpandDimsinputs.tf_op_layer_ExpandDims/ExpandDims/dim:output:0*
T0*
_cloned(*&
_output_shapes
:1(2#
!tf_op_layer_ExpandDims/ExpandDimsп
conv2d/Conv2D/ReadVariableOpReadVariableOp*conv2d_conv2d_readvariableop_conv2d_kernel*&
_output_shapes
:( *
dtype02
conv2d/Conv2D/ReadVariableOp╘
conv2d/Conv2DConv2D*tf_op_layer_ExpandDims/ExpandDims:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
2
conv2d/Conv2D┴
"batch_normalization/ReadVariableOpReadVariableOp<batch_normalization_readvariableop_batch_normalization_gamma*
_output_shapes
: *
dtype02$
"batch_normalization/ReadVariableOp╞
$batch_normalization/ReadVariableOp_1ReadVariableOp=batch_normalization_readvariableop_1_batch_normalization_beta*
_output_shapes
: *
dtype02&
$batch_normalization/ReadVariableOp_1·
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpSbatch_normalization_fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
_output_shapes
: *
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpД
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpYbatch_normalization_fusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
_output_shapes
: *
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1╔
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/Conv2D:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%oГ:*
is_training( 2&
$batch_normalization/FusedBatchNormV3~
re_lu/Relu6Relu6(batch_normalization/FusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2
re_lu/Relu6ъ
)depthwise_conv2d/depthwise/ReadVariableOpReadVariableOpKdepthwise_conv2d_depthwise_readvariableop_depthwise_conv2d_depthwise_kernel*&
_output_shapes
: *
dtype02+
)depthwise_conv2d/depthwise/ReadVariableOpЭ
 depthwise_conv2d/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 depthwise_conv2d/depthwise/Shapeе
(depthwise_conv2d/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2*
(depthwise_conv2d/depthwise/dilation_rate∙
depthwise_conv2d/depthwiseDepthwiseConv2dNativere_lu/Relu6:activations:01depthwise_conv2d/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
2
depthwise_conv2d/depthwise╔
$batch_normalization_1/ReadVariableOpReadVariableOp@batch_normalization_1_readvariableop_batch_normalization_1_gamma*
_output_shapes
: *
dtype02&
$batch_normalization_1/ReadVariableOp╬
&batch_normalization_1/ReadVariableOp_1ReadVariableOpAbatch_normalization_1_readvariableop_1_batch_normalization_1_beta*
_output_shapes
: *
dtype02(
&batch_normalization_1/ReadVariableOp_1В
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_1_fusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
: *
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpМ
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_1_fusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
_output_shapes
: *
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1т
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3#depthwise_conv2d/depthwise:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%oГ:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3Д
re_lu_1/Relu6Relu6*batch_normalization_1/FusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2
re_lu_1/Relu6╖
conv2d_1/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_conv2d_readvariableop_conv2d_1_kernel*&
_output_shapes
:  *
dtype02 
conv2d_1/Conv2D/ReadVariableOp╩
conv2d_1/Conv2DConv2Dre_lu_1/Relu6:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingSAME*
strides
2
conv2d_1/Conv2D╔
$batch_normalization_2/ReadVariableOpReadVariableOp@batch_normalization_2_readvariableop_batch_normalization_2_gamma*
_output_shapes
: *
dtype02&
$batch_normalization_2/ReadVariableOp╬
&batch_normalization_2/ReadVariableOp_1ReadVariableOpAbatch_normalization_2_readvariableop_1_batch_normalization_2_beta*
_output_shapes
: *
dtype02(
&batch_normalization_2/ReadVariableOp_1В
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_2_fusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
: *
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpМ
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_2_fusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
_output_shapes
: *
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1╫
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_1/Conv2D:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%oГ:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3Д
re_lu_2/Relu6Relu6*batch_normalization_2/FusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2
re_lu_2/Relu6Є
+depthwise_conv2d_1/depthwise/ReadVariableOpReadVariableOpOdepthwise_conv2d_1_depthwise_readvariableop_depthwise_conv2d_1_depthwise_kernel*&
_output_shapes
: *
dtype02-
+depthwise_conv2d_1/depthwise/ReadVariableOpб
"depthwise_conv2d_1/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"depthwise_conv2d_1/depthwise/Shapeй
*depthwise_conv2d_1/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2,
*depthwise_conv2d_1/depthwise/dilation_rateБ
depthwise_conv2d_1/depthwiseDepthwiseConv2dNativere_lu_2/Relu6:activations:03depthwise_conv2d_1/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
2
depthwise_conv2d_1/depthwise╔
$batch_normalization_3/ReadVariableOpReadVariableOp@batch_normalization_3_readvariableop_batch_normalization_3_gamma*
_output_shapes
: *
dtype02&
$batch_normalization_3/ReadVariableOp╬
&batch_normalization_3/ReadVariableOp_1ReadVariableOpAbatch_normalization_3_readvariableop_1_batch_normalization_3_beta*
_output_shapes
: *
dtype02(
&batch_normalization_3/ReadVariableOp_1В
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_3_fusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
_output_shapes
: *
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpМ
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_3_fusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
_output_shapes
: *
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ф
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3%depthwise_conv2d_1/depthwise:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%oГ:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3Д
re_lu_3/Relu6Relu6*batch_normalization_3/FusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2
re_lu_3/Relu6╖
conv2d_2/Conv2D/ReadVariableOpReadVariableOp.conv2d_2_conv2d_readvariableop_conv2d_2_kernel*&
_output_shapes
: @*
dtype02 
conv2d_2/Conv2D/ReadVariableOp╩
conv2d_2/Conv2DConv2Dre_lu_3/Relu6:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingSAME*
strides
2
conv2d_2/Conv2D╔
$batch_normalization_4/ReadVariableOpReadVariableOp@batch_normalization_4_readvariableop_batch_normalization_4_gamma*
_output_shapes
:@*
dtype02&
$batch_normalization_4/ReadVariableOp╬
&batch_normalization_4/ReadVariableOp_1ReadVariableOpAbatch_normalization_4_readvariableop_1_batch_normalization_4_beta*
_output_shapes
:@*
dtype02(
&batch_normalization_4/ReadVariableOp_1В
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_4_fusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
_output_shapes
:@*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpМ
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_4_fusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
_output_shapes
:@*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1╫
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_2/Conv2D:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:@:@:@:@:@:*
epsilon%oГ:*
is_training( 2(
&batch_normalization_4/FusedBatchNormV3Д
re_lu_4/Relu6Relu6*batch_normalization_4/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:@2
re_lu_4/Relu6Є
+depthwise_conv2d_2/depthwise/ReadVariableOpReadVariableOpOdepthwise_conv2d_2_depthwise_readvariableop_depthwise_conv2d_2_depthwise_kernel*&
_output_shapes
:@*
dtype02-
+depthwise_conv2d_2/depthwise/ReadVariableOpб
"depthwise_conv2d_2/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @      2$
"depthwise_conv2d_2/depthwise/Shapeй
*depthwise_conv2d_2/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2,
*depthwise_conv2d_2/depthwise/dilation_rateА
depthwise_conv2d_2/depthwiseDepthwiseConv2dNativere_lu_4/Relu6:activations:03depthwise_conv2d_2/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingSAME*
strides
2
depthwise_conv2d_2/depthwise╔
$batch_normalization_5/ReadVariableOpReadVariableOp@batch_normalization_5_readvariableop_batch_normalization_5_gamma*
_output_shapes
:@*
dtype02&
$batch_normalization_5/ReadVariableOp╬
&batch_normalization_5/ReadVariableOp_1ReadVariableOpAbatch_normalization_5_readvariableop_1_batch_normalization_5_beta*
_output_shapes
:@*
dtype02(
&batch_normalization_5/ReadVariableOp_1В
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_5_fusedbatchnormv3_readvariableop_batch_normalization_5_moving_mean*
_output_shapes
:@*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpМ
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_5_fusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance*
_output_shapes
:@*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ф
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3%depthwise_conv2d_2/depthwise:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:@:@:@:@:@:*
epsilon%oГ:*
is_training( 2(
&batch_normalization_5/FusedBatchNormV3Д
re_lu_5/Relu6Relu6*batch_normalization_5/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:@2
re_lu_5/Relu6╕
conv2d_3/Conv2D/ReadVariableOpReadVariableOp.conv2d_3_conv2d_readvariableop_conv2d_3_kernel*'
_output_shapes
:@А*
dtype02 
conv2d_3/Conv2D/ReadVariableOp╦
conv2d_3/Conv2DConv2Dre_lu_5/Relu6:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:А*
paddingSAME*
strides
2
conv2d_3/Conv2D╩
$batch_normalization_6/ReadVariableOpReadVariableOp@batch_normalization_6_readvariableop_batch_normalization_6_gamma*
_output_shapes	
:А*
dtype02&
$batch_normalization_6/ReadVariableOp╧
&batch_normalization_6/ReadVariableOp_1ReadVariableOpAbatch_normalization_6_readvariableop_1_batch_normalization_6_beta*
_output_shapes	
:А*
dtype02(
&batch_normalization_6/ReadVariableOp_1Г
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_6_fusedbatchnormv3_readvariableop_batch_normalization_6_moving_mean*
_output_shapes	
:А*
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpН
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_6_fusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance*
_output_shapes	
:А*
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1▄
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv2d_3/Conv2D:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3:А:А:А:А:А:*
epsilon%oГ:*
is_training( 2(
&batch_normalization_6/FusedBatchNormV3Е
re_lu_6/Relu6Relu6*batch_normalization_6/FusedBatchNormV3:y:0*
T0*'
_output_shapes
:А2
re_lu_6/Relu6є
+depthwise_conv2d_3/depthwise/ReadVariableOpReadVariableOpOdepthwise_conv2d_3_depthwise_readvariableop_depthwise_conv2d_3_depthwise_kernel*'
_output_shapes
:А*
dtype02-
+depthwise_conv2d_3/depthwise/ReadVariableOpб
"depthwise_conv2d_3/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      А      2$
"depthwise_conv2d_3/depthwise/Shapeй
*depthwise_conv2d_3/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2,
*depthwise_conv2d_3/depthwise/dilation_rateБ
depthwise_conv2d_3/depthwiseDepthwiseConv2dNativere_lu_6/Relu6:activations:03depthwise_conv2d_3/depthwise/ReadVariableOp:value:0*
T0*'
_output_shapes
:А*
paddingSAME*
strides
2
depthwise_conv2d_3/depthwise╩
$batch_normalization_7/ReadVariableOpReadVariableOp@batch_normalization_7_readvariableop_batch_normalization_7_gamma*
_output_shapes	
:А*
dtype02&
$batch_normalization_7/ReadVariableOp╧
&batch_normalization_7/ReadVariableOp_1ReadVariableOpAbatch_normalization_7_readvariableop_1_batch_normalization_7_beta*
_output_shapes	
:А*
dtype02(
&batch_normalization_7/ReadVariableOp_1Г
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_7_fusedbatchnormv3_readvariableop_batch_normalization_7_moving_mean*
_output_shapes	
:А*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpН
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_7_fusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance*
_output_shapes	
:А*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1щ
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3%depthwise_conv2d_3/depthwise:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3:А:А:А:А:А:*
epsilon%oГ:*
is_training( 2(
&batch_normalization_7/FusedBatchNormV3Е
re_lu_7/Relu6Relu6*batch_normalization_7/FusedBatchNormV3:y:0*
T0*'
_output_shapes
:А2
re_lu_7/Relu6╣
conv2d_4/Conv2D/ReadVariableOpReadVariableOp.conv2d_4_conv2d_readvariableop_conv2d_4_kernel*(
_output_shapes
:АА*
dtype02 
conv2d_4/Conv2D/ReadVariableOp╦
conv2d_4/Conv2DConv2Dre_lu_7/Relu6:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:А*
paddingSAME*
strides
2
conv2d_4/Conv2D╩
$batch_normalization_8/ReadVariableOpReadVariableOp@batch_normalization_8_readvariableop_batch_normalization_8_gamma*
_output_shapes	
:А*
dtype02&
$batch_normalization_8/ReadVariableOp╧
&batch_normalization_8/ReadVariableOp_1ReadVariableOpAbatch_normalization_8_readvariableop_1_batch_normalization_8_beta*
_output_shapes	
:А*
dtype02(
&batch_normalization_8/ReadVariableOp_1Г
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_8_fusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean*
_output_shapes	
:А*
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpН
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_8_fusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance*
_output_shapes	
:А*
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1▄
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3conv2d_4/Conv2D:output:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3:А:А:А:А:А:*
epsilon%oГ:*
is_training( 2(
&batch_normalization_8/FusedBatchNormV3Е
re_lu_8/Relu6Relu6*batch_normalization_8/FusedBatchNormV3:y:0*
T0*'
_output_shapes
:А2
re_lu_8/Relu6│
/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      21
/global_average_pooling2d/Mean/reduction_indices╟
global_average_pooling2d/MeanMeanre_lu_8/Relu6:activations:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*
_output_shapes
:	А2
global_average_pooling2d/MeanВ
dropout/IdentityIdentity&global_average_pooling2d/Mean:output:0*
T0*
_output_shapes
:	А2
dropout/Identityд
dense/MatMul/ReadVariableOpReadVariableOp(dense_matmul_readvariableop_dense_kernel*
_output_shapes
:	А*
dtype02
dense/MatMul/ReadVariableOpП
dense/MatMulMatMuldropout/Identity:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense/MatMulа
dense/BiasAdd/ReadVariableOpReadVariableOp'dense_biasadd_readvariableop_dense_bias*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOpР
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
identityIdentity:output:0*▀
_input_shapes═
╩:1(::::::::::::::::::::::::::::::::::::::::::::::::S O
+
_output_shapes
:         1(
 
_user_specified_nameinputs
ю
у
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2243

inputs.
*readvariableop_batch_normalization_6_gamma/
+readvariableop_1_batch_normalization_6_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_6_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance
identityИИ
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_6_gamma*
_output_shapes	
:А*
dtype02
ReadVariableOpН
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_6_beta*
_output_shapes	
:А*
dtype02
ReadVariableOp_1┴
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_6_moving_mean*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOp╦
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╞
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3:А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3h
IdentityIdentityFusedBatchNormV3:y:0*
T0*'
_output_shapes
:А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:А:::::O K
'
_output_shapes
:А
 
_user_specified_nameinputs
╜и
У
F__inference_functional_1_layer_call_and_return_conditional_losses_2675

inputs
conv2d_conv2d_kernel1
-batch_normalization_batch_normalization_gamma0
,batch_normalization_batch_normalization_beta7
3batch_normalization_batch_normalization_moving_mean;
7batch_normalization_batch_normalization_moving_variance6
2depthwise_conv2d_depthwise_conv2d_depthwise_kernel5
1batch_normalization_1_batch_normalization_1_gamma4
0batch_normalization_1_batch_normalization_1_beta;
7batch_normalization_1_batch_normalization_1_moving_mean?
;batch_normalization_1_batch_normalization_1_moving_variance
conv2d_1_conv2d_1_kernel5
1batch_normalization_2_batch_normalization_2_gamma4
0batch_normalization_2_batch_normalization_2_beta;
7batch_normalization_2_batch_normalization_2_moving_mean?
;batch_normalization_2_batch_normalization_2_moving_variance:
6depthwise_conv2d_1_depthwise_conv2d_1_depthwise_kernel5
1batch_normalization_3_batch_normalization_3_gamma4
0batch_normalization_3_batch_normalization_3_beta;
7batch_normalization_3_batch_normalization_3_moving_mean?
;batch_normalization_3_batch_normalization_3_moving_variance
conv2d_2_conv2d_2_kernel5
1batch_normalization_4_batch_normalization_4_gamma4
0batch_normalization_4_batch_normalization_4_beta;
7batch_normalization_4_batch_normalization_4_moving_mean?
;batch_normalization_4_batch_normalization_4_moving_variance:
6depthwise_conv2d_2_depthwise_conv2d_2_depthwise_kernel5
1batch_normalization_5_batch_normalization_5_gamma4
0batch_normalization_5_batch_normalization_5_beta;
7batch_normalization_5_batch_normalization_5_moving_mean?
;batch_normalization_5_batch_normalization_5_moving_variance
conv2d_3_conv2d_3_kernel5
1batch_normalization_6_batch_normalization_6_gamma4
0batch_normalization_6_batch_normalization_6_beta;
7batch_normalization_6_batch_normalization_6_moving_mean?
;batch_normalization_6_batch_normalization_6_moving_variance:
6depthwise_conv2d_3_depthwise_conv2d_3_depthwise_kernel5
1batch_normalization_7_batch_normalization_7_gamma4
0batch_normalization_7_batch_normalization_7_beta;
7batch_normalization_7_batch_normalization_7_moving_mean?
;batch_normalization_7_batch_normalization_7_moving_variance
conv2d_4_conv2d_4_kernel5
1batch_normalization_8_batch_normalization_8_gamma4
0batch_normalization_8_batch_normalization_8_beta;
7batch_normalization_8_batch_normalization_8_moving_mean?
;batch_normalization_8_batch_normalization_8_moving_variance
dense_dense_kernel
dense_dense_bias
identityИв+batch_normalization/StatefulPartitionedCallв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallв-batch_normalization_3/StatefulPartitionedCallв-batch_normalization_4/StatefulPartitionedCallв-batch_normalization_5/StatefulPartitionedCallв-batch_normalization_6/StatefulPartitionedCallв-batch_normalization_7/StatefulPartitionedCallв-batch_normalization_8/StatefulPartitionedCallвconv2d/StatefulPartitionedCallв conv2d_1/StatefulPartitionedCallв conv2d_2/StatefulPartitionedCallв conv2d_3/StatefulPartitionedCallв conv2d_4/StatefulPartitionedCallвdense/StatefulPartitionedCallв(depthwise_conv2d/StatefulPartitionedCallв*depthwise_conv2d_1/StatefulPartitionedCallв*depthwise_conv2d_2/StatefulPartitionedCallв*depthwise_conv2d_3/StatefulPartitionedCallвdropout/StatefulPartitionedCall√
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
GPU 2J 8В *Y
fTRR
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_16752(
&tf_op_layer_ExpandDims/PartitionedCallж
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
GPU 2J 8В *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_16902 
conv2d/StatefulPartitionedCall■
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0-batch_normalization_batch_normalization_gamma,batch_normalization_batch_normalization_beta3batch_normalization_batch_normalization_moving_mean7batch_normalization_batch_normalization_moving_variance*
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
GPU 2J 8В *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_17182-
+batch_normalization/StatefulPartitionedCallЎ
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
GPU 2J 8В *H
fCRA
?__inference_re_lu_layer_call_and_return_conditional_losses_17652
re_lu/PartitionedCall╨
(depthwise_conv2d/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:02depthwise_conv2d_depthwise_conv2d_depthwise_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_depthwise_conv2d_layer_call_and_return_conditional_losses_8282*
(depthwise_conv2d/StatefulPartitionedCallЮ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall1depthwise_conv2d/StatefulPartitionedCall:output:01batch_normalization_1_batch_normalization_1_gamma0batch_normalization_1_batch_normalization_1_beta7batch_normalization_1_batch_normalization_1_moving_mean;batch_normalization_1_batch_normalization_1_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_17982/
-batch_normalization_1/StatefulPartitionedCall■
re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_re_lu_1_layer_call_and_return_conditional_losses_18452
re_lu_1/PartitionedCallб
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall re_lu_1/PartitionedCall:output:0conv2d_1_conv2d_1_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_18602"
 conv2d_1/StatefulPartitionedCallЦ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:01batch_normalization_2_batch_normalization_2_gamma0batch_normalization_2_batch_normalization_2_beta7batch_normalization_2_batch_normalization_2_moving_mean;batch_normalization_2_batch_normalization_2_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_18872/
-batch_normalization_2/StatefulPartitionedCall■
re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_re_lu_2_layer_call_and_return_conditional_losses_19342
re_lu_2/PartitionedCall▌
*depthwise_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall re_lu_2/PartitionedCall:output:06depthwise_conv2d_1_depthwise_conv2d_1_depthwise_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_depthwise_conv2d_1_layer_call_and_return_conditional_losses_10382,
*depthwise_conv2d_1/StatefulPartitionedCallа
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall3depthwise_conv2d_1/StatefulPartitionedCall:output:01batch_normalization_3_batch_normalization_3_gamma0batch_normalization_3_batch_normalization_3_beta7batch_normalization_3_batch_normalization_3_moving_mean;batch_normalization_3_batch_normalization_3_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_19672/
-batch_normalization_3/StatefulPartitionedCall■
re_lu_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_re_lu_3_layer_call_and_return_conditional_losses_20142
re_lu_3/PartitionedCallб
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall re_lu_3/PartitionedCall:output:0conv2d_2_conv2d_2_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_20292"
 conv2d_2/StatefulPartitionedCallЦ
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:01batch_normalization_4_batch_normalization_4_gamma0batch_normalization_4_batch_normalization_4_beta7batch_normalization_4_batch_normalization_4_moving_mean;batch_normalization_4_batch_normalization_4_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_20562/
-batch_normalization_4/StatefulPartitionedCall■
re_lu_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_re_lu_4_layer_call_and_return_conditional_losses_21032
re_lu_4/PartitionedCall▌
*depthwise_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:06depthwise_conv2d_2_depthwise_conv2d_2_depthwise_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_depthwise_conv2d_2_layer_call_and_return_conditional_losses_12482,
*depthwise_conv2d_2/StatefulPartitionedCallа
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall3depthwise_conv2d_2/StatefulPartitionedCall:output:01batch_normalization_5_batch_normalization_5_gamma0batch_normalization_5_batch_normalization_5_beta7batch_normalization_5_batch_normalization_5_moving_mean;batch_normalization_5_batch_normalization_5_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_21362/
-batch_normalization_5/StatefulPartitionedCall■
re_lu_5/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_re_lu_5_layer_call_and_return_conditional_losses_21832
re_lu_5/PartitionedCallв
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_5/PartitionedCall:output:0conv2d_3_conv2d_3_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:А*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_21982"
 conv2d_3/StatefulPartitionedCallЧ
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:01batch_normalization_6_batch_normalization_6_gamma0batch_normalization_6_batch_normalization_6_beta7batch_normalization_6_batch_normalization_6_moving_mean;batch_normalization_6_batch_normalization_6_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_22252/
-batch_normalization_6/StatefulPartitionedCall 
re_lu_6/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_re_lu_6_layer_call_and_return_conditional_losses_22722
re_lu_6/PartitionedCall▐
*depthwise_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_6/PartitionedCall:output:06depthwise_conv2d_3_depthwise_conv2d_3_depthwise_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:А*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_depthwise_conv2d_3_layer_call_and_return_conditional_losses_14582,
*depthwise_conv2d_3/StatefulPartitionedCallб
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall3depthwise_conv2d_3/StatefulPartitionedCall:output:01batch_normalization_7_batch_normalization_7_gamma0batch_normalization_7_batch_normalization_7_beta7batch_normalization_7_batch_normalization_7_moving_mean;batch_normalization_7_batch_normalization_7_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_23052/
-batch_normalization_7/StatefulPartitionedCall 
re_lu_7/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_re_lu_7_layer_call_and_return_conditional_losses_23522
re_lu_7/PartitionedCallв
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall re_lu_7/PartitionedCall:output:0conv2d_4_conv2d_4_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:А*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_23672"
 conv2d_4/StatefulPartitionedCallЧ
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:01batch_normalization_8_batch_normalization_8_gamma0batch_normalization_8_batch_normalization_8_beta7batch_normalization_8_batch_normalization_8_moving_mean;batch_normalization_8_batch_normalization_8_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_23942/
-batch_normalization_8/StatefulPartitionedCall 
re_lu_8/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_re_lu_8_layer_call_and_return_conditional_losses_24412
re_lu_8/PartitionedCallФ
(global_average_pooling2d/PartitionedCallPartitionedCall re_lu_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_16622*
(global_average_pooling2d/PartitionedCallК
dropout/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_24672!
dropout/StatefulPartitionedCallж
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
GPU 2J 8В *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_24952
dense/StatefulPartitionedCall└
IdentityIdentity&dense/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall^dense/StatefulPartitionedCall)^depthwise_conv2d/StatefulPartitionedCall+^depthwise_conv2d_1/StatefulPartitionedCall+^depthwise_conv2d_2/StatefulPartitionedCall+^depthwise_conv2d_3/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*▀
_input_shapes═
╩:1(:::::::::::::::::::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2T
(depthwise_conv2d/StatefulPartitionedCall(depthwise_conv2d/StatefulPartitionedCall2X
*depthwise_conv2d_1/StatefulPartitionedCall*depthwise_conv2d_1/StatefulPartitionedCall2X
*depthwise_conv2d_2/StatefulPartitionedCall*depthwise_conv2d_2/StatefulPartitionedCall2X
*depthwise_conv2d_3/StatefulPartitionedCall*depthwise_conv2d_3/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall:S O
+
_output_shapes
:         1(
 
_user_specified_nameinputs
ў
З
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_4836

inputs.
*readvariableop_batch_normalization_7_gamma/
+readvariableop_1_batch_normalization_7_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_7_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance
identityИвAssignNewValueвAssignNewValue_1И
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_7_gamma*
_output_shapes	
:А*
dtype02
ReadVariableOpН
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_7_beta*
_output_shapes	
:А*
dtype02
ReadVariableOp_1┴
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_7_moving_mean*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOp╦
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1я
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3▒
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_7_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_7/moving_mean*
_output_shapes
 *
dtype02
AssignNewValue╟
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_7/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1з
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,                           А::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
ў
З
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1402

inputs.
*readvariableop_batch_normalization_6_gamma/
+readvariableop_1_batch_normalization_6_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_6_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance
identityИвAssignNewValueвAssignNewValue_1И
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_6_gamma*
_output_shapes	
:А*
dtype02
ReadVariableOpН
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_6_beta*
_output_shapes	
:А*
dtype02
ReadVariableOp_1┴
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_6_moving_mean*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOp╦
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1я
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3▒
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_6_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_6/moving_mean*
_output_shapes
 *
dtype02
AssignNewValue╟
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_6/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1з
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,                           А::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
М	
и
I__inference_depthwise_conv2d_layer_call_and_return_conditional_losses_816

inputs>
:depthwise_readvariableop_depthwise_conv2d_depthwise_kernel
identityИ╖
depthwise/ReadVariableOpReadVariableOp:depthwise_readvariableop_depthwise_conv2d_depthwise_kernel*&
_output_shapes
: *
dtype02
depthwise/ReadVariableOp{
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2
depthwise/ShapeГ
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
depthwise/dilation_rate╬
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingVALID*
strides
2
	depthwiseА
IdentityIdentitydepthwise:output:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+                            ::i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
╧
у
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_4302

inputs.
*readvariableop_batch_normalization_3_gamma/
+readvariableop_1_batch_normalization_3_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_3_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance
identityИЗ
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_3_gamma*
_output_shapes
: *
dtype02
ReadVariableOpМ
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_3_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1└
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp╩
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            :::::i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
│
]
A__inference_re_lu_6_layer_call_and_return_conditional_losses_4759

inputs
identityQ
Relu6Relu6inputs*
T0*'
_output_shapes
:А2
Relu6g
IdentityIdentityRelu6:activations:0*
T0*'
_output_shapes
:А2

Identity"
identityIdentity:output:0*&
_input_shapes
:А:O K
'
_output_shapes
:А
 
_user_specified_nameinputs
т
у
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1816

inputs.
*readvariableop_batch_normalization_1_gamma/
+readvariableop_1_batch_normalization_1_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_1_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance
identityИЗ
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_1_gamma*
_output_shapes
: *
dtype02
ReadVariableOpМ
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_1_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1└
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp╩
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1┴
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3g
IdentityIdentityFusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": :::::N J
&
_output_shapes
: 
 
_user_specified_nameinputs
ю
у
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_4736

inputs.
*readvariableop_batch_normalization_6_gamma/
+readvariableop_1_batch_normalization_6_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_6_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance
identityИИ
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_6_gamma*
_output_shapes	
:А*
dtype02
ReadVariableOpН
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_6_beta*
_output_shapes	
:А*
dtype02
ReadVariableOp_1┴
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_6_moving_mean*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOp╦
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╞
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3:А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3h
IdentityIdentityFusedBatchNormV3:y:0*
T0*'
_output_shapes
:А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:А:::::O K
'
_output_shapes
:А
 
_user_specified_nameinputs
╚
u
'__inference_conv2d_4_layer_call_fn_4895

inputs
conv2d_4_kernel
identityИвStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_4_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:А*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_23672
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:А2

Identity"
identityIdentity:output:0**
_input_shapes
:А:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:А
 
_user_specified_nameinputs
Ю
Q
5__inference_tf_op_layer_ExpandDims_layer_call_fn_3886

inputs
identity═
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
GPU 2J 8В *Y
fTRR
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_16752
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
╧
М
B__inference_conv2d_4_layer_call_and_return_conditional_losses_2367

inputs)
%conv2d_readvariableop_conv2d_4_kernel
identityИЮ
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_4_kernel*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOpЫ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:А*
paddingSAME*
strides
2
Conv2Dc
IdentityIdentityConv2D:output:0*
T0*'
_output_shapes
:А2

Identity"
identityIdentity:output:0**
_input_shapes
:А::O K
'
_output_shapes
:А
 
_user_specified_nameinputs
╜
┘
M__inference_batch_normalization_layer_call_and_return_conditional_losses_3935

inputs,
(readvariableop_batch_normalization_gamma-
)readvariableop_1_batch_normalization_betaC
?fusedbatchnormv3_readvariableop_batch_normalization_moving_meanI
Efusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance
identityИЕ
ReadVariableOpReadVariableOp(readvariableop_batch_normalization_gamma*
_output_shapes
: *
dtype02
ReadVariableOpК
ReadVariableOp_1ReadVariableOp)readvariableop_1_batch_normalization_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1╛
FusedBatchNormV3/ReadVariableOpReadVariableOp?fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp╚
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpEfusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            :::::i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
ю
у
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_4800

inputs.
*readvariableop_batch_normalization_7_gamma/
+readvariableop_1_batch_normalization_7_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_7_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance
identityИИ
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_7_gamma*
_output_shapes	
:А*
dtype02
ReadVariableOpН
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_7_beta*
_output_shapes	
:А*
dtype02
ReadVariableOp_1┴
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_7_moving_mean*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOp╦
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╞
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3:А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3h
IdentityIdentityFusedBatchNormV3:y:0*
T0*'
_output_shapes
:А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:А:::::O K
'
_output_shapes
:А
 
_user_specified_nameinputs
с
А
4__inference_batch_normalization_5_layer_call_fn_4569

inputs
batch_normalization_5_gamma
batch_normalization_5_beta%
!batch_normalization_5_moving_mean)
%batch_normalization_5_moving_variance
identityИвStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_5_gammabatch_normalization_5_beta!batch_normalization_5_moving_mean%batch_normalization_5_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_21542
StatefulPartitionedCallН
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*5
_input_shapes$
":@::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:@
 
_user_specified_nameinputs
│
]
A__inference_re_lu_6_layer_call_and_return_conditional_losses_2272

inputs
identityQ
Relu6Relu6inputs*
T0*'
_output_shapes
:А2
Relu6g
IdentityIdentityRelu6:activations:0*
T0*'
_output_shapes
:А2

Identity"
identityIdentity:output:0*&
_input_shapes
:А:O K
'
_output_shapes
:А
 
_user_specified_nameinputs
╠
М
B__inference_conv2d_3_layer_call_and_return_conditional_losses_2198

inputs)
%conv2d_readvariableop_conv2d_3_kernel
identityИЭ
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_3_kernel*'
_output_shapes
:@А*
dtype02
Conv2D/ReadVariableOpЫ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:А*
paddingSAME*
strides
2
Conv2Dc
IdentityIdentityConv2D:output:0*
T0*'
_output_shapes
:А2

Identity"
identityIdentity:output:0*)
_input_shapes
:@::N J
&
_output_shapes
:@
 
_user_specified_nameinputs
 
З
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1798

inputs.
*readvariableop_batch_normalization_1_gamma/
+readvariableop_1_batch_normalization_1_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_1_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance
identityИвAssignNewValueвAssignNewValue_1З
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_1_gamma*
_output_shapes
: *
dtype02
ReadVariableOpМ
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_1_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1└
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp╩
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╧
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3▒
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_1_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_1/moving_mean*
_output_shapes
 *
dtype02
AssignNewValue╟
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_1/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1Л
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:N J
&
_output_shapes
: 
 
_user_specified_nameinputs
п
]
A__inference_re_lu_2_layer_call_and_return_conditional_losses_4261

inputs
identityP
Relu6Relu6inputs*
T0*&
_output_shapes
: 2
Relu6f
IdentityIdentityRelu6:activations:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*%
_input_shapes
: :N J
&
_output_shapes
: 
 
_user_specified_nameinputs
х
¤
M__inference_batch_normalization_layer_call_and_return_conditional_losses_3971

inputs,
(readvariableop_batch_normalization_gamma-
)readvariableop_1_batch_normalization_betaC
?fusedbatchnormv3_readvariableop_batch_normalization_moving_meanI
Efusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance
identityИвAssignNewValueвAssignNewValue_1Е
ReadVariableOpReadVariableOp(readvariableop_batch_normalization_gamma*
_output_shapes
: *
dtype02
ReadVariableOpК
ReadVariableOp_1ReadVariableOp)readvariableop_1_batch_normalization_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1╛
FusedBatchNormV3/ReadVariableOpReadVariableOp?fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp╚
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpEfusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╧
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3н
AssignNewValueAssignVariableOp?fusedbatchnormv3_readvariableop_batch_normalization_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization/moving_mean*
_output_shapes
 *
dtype02
AssignNewValue├
AssignNewValue_1AssignVariableOpEfusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*X
_classN
LJloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1Л
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
╔
М
B__inference_conv2d_2_layer_call_and_return_conditional_losses_4391

inputs)
%conv2d_readvariableop_conv2d_2_kernel
identityИЬ
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_2_kernel*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOpЪ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingSAME*
strides
2
Conv2Db
IdentityIdentityConv2D:output:0*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*)
_input_shapes
: ::N J
&
_output_shapes
: 
 
_user_specified_nameinputs
╕
n
R__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_1662

inputs
identityБ
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:                  2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:                  2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
д
_
A__inference_dropout_layer_call_and_return_conditional_losses_2472

inputs

identity_1R
IdentityIdentityinputs*
T0*
_output_shapes
:	А2

Identitya

Identity_1IdentityIdentity:output:0*
T0*
_output_shapes
:	А2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
:	А:G C

_output_shapes
:	А
 
_user_specified_nameinputs
Т	
н
L__inference_depthwise_conv2d_2_layer_call_and_return_conditional_losses_1236

inputs@
<depthwise_readvariableop_depthwise_conv2d_2_depthwise_kernel
identityИ╣
depthwise/ReadVariableOpReadVariableOp<depthwise_readvariableop_depthwise_conv2d_2_depthwise_kernel*&
_output_shapes
:@*
dtype02
depthwise/ReadVariableOp{
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @      2
depthwise/ShapeГ
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
depthwise/dilation_rate═
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @*
paddingSAME*
strides
2
	depthwiseА
IdentityIdentitydepthwise:output:0*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+                           @::i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
н
[
?__inference_re_lu_layer_call_and_return_conditional_losses_1765

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
М
B
&__inference_re_lu_7_layer_call_fn_4882

inputs
identity┐
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_re_lu_7_layer_call_and_return_conditional_losses_23522
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:А2

Identity"
identityIdentity:output:0*&
_input_shapes
:А:O K
'
_output_shapes
:А
 
_user_specified_nameinputs
╦	
А
4__inference_batch_normalization_3_layer_call_fn_4311

inputs
batch_normalization_3_gamma
batch_normalization_3_beta%
!batch_normalization_3_moving_mean)
%batch_normalization_3_moving_variance
identityИвStatefulPartitionedCallК
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_3_gammabatch_normalization_3_beta!batch_normalization_3_moving_mean%batch_normalization_3_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_11002
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
д
_
A__inference_dropout_layer_call_and_return_conditional_losses_5030

inputs

identity_1R
IdentityIdentityinputs*
T0*
_output_shapes
:	А2

Identitya

Identity_1IdentityIdentity:output:0*
T0*
_output_shapes
:	А2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
:	А:G C

_output_shapes
:	А
 
_user_specified_nameinputs
у
А
4__inference_batch_normalization_6_layer_call_fn_4745

inputs
batch_normalization_6_gamma
batch_normalization_6_beta%
!batch_normalization_6_moving_mean)
%batch_normalization_6_moving_variance
identityИвStatefulPartitionedCallЁ
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_6_gammabatch_normalization_6_beta!batch_normalization_6_moving_mean%batch_normalization_6_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_22252
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:А::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:А
 
_user_specified_nameinputs
╤	
А
4__inference_batch_normalization_7_layer_call_fn_4872

inputs
batch_normalization_7_gamma
batch_normalization_7_beta%
!batch_normalization_7_moving_mean)
%batch_normalization_7_moving_variance
identityИвStatefulPartitionedCallН
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_7_gammabatch_normalization_7_beta!batch_normalization_7_moving_mean%batch_normalization_7_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_15472
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,                           А::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
Ч	
н
L__inference_depthwise_conv2d_3_layer_call_and_return_conditional_losses_1446

inputs@
<depthwise_readvariableop_depthwise_conv2d_3_depthwise_kernel
identityИ║
depthwise/ReadVariableOpReadVariableOp<depthwise_readvariableop_depthwise_conv2d_3_depthwise_kernel*'
_output_shapes
:А*
dtype02
depthwise/ReadVariableOp{
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      А      2
depthwise/ShapeГ
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
depthwise/dilation_rate╬
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           А*
paddingSAME*
strides
2
	depthwiseБ
IdentityIdentitydepthwise:output:0*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:,                           А::j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
Ж
S
7__inference_global_average_pooling2d_layer_call_fn_1665

inputs
identity┘
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:                  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_16622
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:                  2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
М
B
&__inference_re_lu_6_layer_call_fn_4764

inputs
identity┐
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_re_lu_6_layer_call_and_return_conditional_losses_22722
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:А2

Identity"
identityIdentity:output:0*&
_input_shapes
:А:O K
'
_output_shapes
:А
 
_user_specified_nameinputs
 
З
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2136

inputs.
*readvariableop_batch_normalization_5_gamma/
+readvariableop_1_batch_normalization_5_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_5_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance
identityИвAssignNewValueвAssignNewValue_1З
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_5_gamma*
_output_shapes
:@*
dtype02
ReadVariableOpМ
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_5_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1└
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_5_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp╩
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╧
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3▒
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_5_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_5/moving_mean*
_output_shapes
 *
dtype02
AssignNewValue╟
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_5/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1Л
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*5
_input_shapes$
":@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:N J
&
_output_shapes
:@
 
_user_specified_nameinputs
╨
┘
M__inference_batch_normalization_layer_call_and_return_conditional_losses_3989

inputs,
(readvariableop_batch_normalization_gamma-
)readvariableop_1_batch_normalization_betaC
?fusedbatchnormv3_readvariableop_batch_normalization_moving_meanI
Efusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance
identityИЕ
ReadVariableOpReadVariableOp(readvariableop_batch_normalization_gamma*
_output_shapes
: *
dtype02
ReadVariableOpК
ReadVariableOp_1ReadVariableOp)readvariableop_1_batch_normalization_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1╛
FusedBatchNormV3/ReadVariableOpReadVariableOp?fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp╚
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpEfusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1┴
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%oГ:*
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
т
у
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_4433

inputs.
*readvariableop_batch_normalization_4_gamma/
+readvariableop_1_batch_normalization_4_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_4_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance
identityИЗ
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_4_gamma*
_output_shapes
:@*
dtype02
ReadVariableOpМ
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_4_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1└
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp╩
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1┴
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3g
IdentityIdentityFusedBatchNormV3:y:0*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*5
_input_shapes$
":@:::::N J
&
_output_shapes
:@
 
_user_specified_nameinputs
И
B
&__inference_re_lu_5_layer_call_fn_4633

inputs
identity╛
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_re_lu_5_layer_call_and_return_conditional_losses_21832
PartitionedCallk
IdentityIdentityPartitionedCall:output:0*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*%
_input_shapes
:@:N J
&
_output_shapes
:@
 
_user_specified_nameinputs
Т	
н
L__inference_depthwise_conv2d_2_layer_call_and_return_conditional_losses_1248

inputs@
<depthwise_readvariableop_depthwise_conv2d_2_depthwise_kernel
identityИ╣
depthwise/ReadVariableOpReadVariableOp<depthwise_readvariableop_depthwise_conv2d_2_depthwise_kernel*&
_output_shapes
:@*
dtype02
depthwise/ReadVariableOp{
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @      2
depthwise/ShapeГ
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
depthwise/dilation_rate═
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @*
paddingSAME*
strides
2
	depthwiseА
IdentityIdentitydepthwise:output:0*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+                           @::i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
╔
М
B__inference_conv2d_2_layer_call_and_return_conditional_losses_2029

inputs)
%conv2d_readvariableop_conv2d_2_kernel
identityИЬ
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_2_kernel*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOpЪ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingSAME*
strides
2
Conv2Db
IdentityIdentityConv2D:output:0*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*)
_input_shapes
: ::N J
&
_output_shapes
: 
 
_user_specified_nameinputs
ў
З
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_4967

inputs.
*readvariableop_batch_normalization_8_gamma/
+readvariableop_1_batch_normalization_8_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_8_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance
identityИвAssignNewValueвAssignNewValue_1И
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_8_gamma*
_output_shapes	
:А*
dtype02
ReadVariableOpН
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_8_beta*
_output_shapes	
:А*
dtype02
ReadVariableOp_1┴
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOp╦
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1я
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3▒
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_8_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_8/moving_mean*
_output_shapes
 *
dtype02
AssignNewValue╟
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_8/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1з
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,                           А::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
╝
╪
L__inference_batch_normalization_layer_call_and_return_conditional_losses_799

inputs,
(readvariableop_batch_normalization_gamma-
)readvariableop_1_batch_normalization_betaC
?fusedbatchnormv3_readvariableop_batch_normalization_moving_meanI
Efusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance
identityИЕ
ReadVariableOpReadVariableOp(readvariableop_batch_normalization_gamma*
_output_shapes
: *
dtype02
ReadVariableOpК
ReadVariableOp_1ReadVariableOp)readvariableop_1_batch_normalization_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1╛
FusedBatchNormV3/ReadVariableOpReadVariableOp?fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp╚
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpEfusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            :::::i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
█
у
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1547

inputs.
*readvariableop_batch_normalization_7_gamma/
+readvariableop_1_batch_normalization_7_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_7_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance
identityИИ
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_7_gamma*
_output_shapes	
:А*
dtype02
ReadVariableOpН
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_7_beta*
_output_shapes	
:А*
dtype02
ReadVariableOp_1┴
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_7_moving_mean*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOp╦
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1с
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Г
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,                           А:::::j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
═	
А
4__inference_batch_normalization_5_layer_call_fn_4623

inputs
batch_normalization_5_gamma
batch_normalization_5_beta%
!batch_normalization_5_moving_mean)
%batch_normalization_5_moving_variance
identityИвStatefulPartitionedCallМ
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_5_gammabatch_normalization_5_beta!batch_normalization_5_moving_mean%batch_normalization_5_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_13372
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
М
B
&__inference_re_lu_8_layer_call_fn_5013

inputs
identity┐
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_re_lu_8_layer_call_and_return_conditional_losses_24412
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:А2

Identity"
identityIdentity:output:0*&
_input_shapes
:А:O K
'
_output_shapes
:А
 
_user_specified_nameinputs
И
B
&__inference_re_lu_2_layer_call_fn_4266

inputs
identity╛
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_re_lu_2_layer_call_and_return_conditional_losses_19342
PartitionedCallk
IdentityIdentityPartitionedCall:output:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*%
_input_shapes
: :N J
&
_output_shapes
: 
 
_user_specified_nameinputs
т
у
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2074

inputs.
*readvariableop_batch_normalization_4_gamma/
+readvariableop_1_batch_normalization_4_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_4_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance
identityИЗ
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_4_gamma*
_output_shapes
:@*
dtype02
ReadVariableOpМ
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_4_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1└
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp╩
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1┴
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3g
IdentityIdentityFusedBatchNormV3:y:0*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*5
_input_shapes$
":@:::::N J
&
_output_shapes
:@
 
_user_specified_nameinputs
Ж 
▓
+__inference_functional_1_layer_call_fn_3823

inputs
conv2d_kernel
batch_normalization_gamma
batch_normalization_beta#
batch_normalization_moving_mean'
#batch_normalization_moving_variance%
!depthwise_conv2d_depthwise_kernel
batch_normalization_1_gamma
batch_normalization_1_beta%
!batch_normalization_1_moving_mean)
%batch_normalization_1_moving_variance
conv2d_1_kernel
batch_normalization_2_gamma
batch_normalization_2_beta%
!batch_normalization_2_moving_mean)
%batch_normalization_2_moving_variance'
#depthwise_conv2d_1_depthwise_kernel
batch_normalization_3_gamma
batch_normalization_3_beta%
!batch_normalization_3_moving_mean)
%batch_normalization_3_moving_variance
conv2d_2_kernel
batch_normalization_4_gamma
batch_normalization_4_beta%
!batch_normalization_4_moving_mean)
%batch_normalization_4_moving_variance'
#depthwise_conv2d_2_depthwise_kernel
batch_normalization_5_gamma
batch_normalization_5_beta%
!batch_normalization_5_moving_mean)
%batch_normalization_5_moving_variance
conv2d_3_kernel
batch_normalization_6_gamma
batch_normalization_6_beta%
!batch_normalization_6_moving_mean)
%batch_normalization_6_moving_variance'
#depthwise_conv2d_3_depthwise_kernel
batch_normalization_7_gamma
batch_normalization_7_beta%
!batch_normalization_7_moving_mean)
%batch_normalization_7_moving_variance
conv2d_4_kernel
batch_normalization_8_gamma
batch_normalization_8_beta%
!batch_normalization_8_moving_mean)
%batch_normalization_8_moving_variance
dense_kernel

dense_bias
identityИвStatefulPartitionedCall│
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_kernelbatch_normalization_gammabatch_normalization_betabatch_normalization_moving_mean#batch_normalization_moving_variance!depthwise_conv2d_depthwise_kernelbatch_normalization_1_gammabatch_normalization_1_beta!batch_normalization_1_moving_mean%batch_normalization_1_moving_varianceconv2d_1_kernelbatch_normalization_2_gammabatch_normalization_2_beta!batch_normalization_2_moving_mean%batch_normalization_2_moving_variance#depthwise_conv2d_1_depthwise_kernelbatch_normalization_3_gammabatch_normalization_3_beta!batch_normalization_3_moving_mean%batch_normalization_3_moving_varianceconv2d_2_kernelbatch_normalization_4_gammabatch_normalization_4_beta!batch_normalization_4_moving_mean%batch_normalization_4_moving_variance#depthwise_conv2d_2_depthwise_kernelbatch_normalization_5_gammabatch_normalization_5_beta!batch_normalization_5_moving_mean%batch_normalization_5_moving_varianceconv2d_3_kernelbatch_normalization_6_gammabatch_normalization_6_beta!batch_normalization_6_moving_mean%batch_normalization_6_moving_variance#depthwise_conv2d_3_depthwise_kernelbatch_normalization_7_gammabatch_normalization_7_beta!batch_normalization_7_moving_mean%batch_normalization_7_moving_varianceconv2d_4_kernelbatch_normalization_8_gammabatch_normalization_8_beta!batch_normalization_8_moving_mean%batch_normalization_8_moving_variancedense_kernel
dense_bias*;
Tin4
220*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*?
_read_only_resource_inputs!
 !$%&)*+./*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_functional_1_layer_call_and_return_conditional_losses_26752
StatefulPartitionedCallЕ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*ш
_input_shapes╓
╙:         1(:::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         1(
 
_user_specified_nameinputs
╠	
А
4__inference_batch_normalization_1_layer_call_fn_4125

inputs
batch_normalization_1_gamma
batch_normalization_1_beta%
!batch_normalization_1_moving_mean)
%batch_normalization_1_moving_variance
identityИвStatefulPartitionedCallЛ
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_1_gammabatch_normalization_1_beta!batch_normalization_1_moving_mean%batch_normalization_1_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_batch_normalization_1_layer_call_and_return_conditional_losses_9172
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
╨
┘
M__inference_batch_normalization_layer_call_and_return_conditional_losses_1736

inputs,
(readvariableop_batch_normalization_gamma-
)readvariableop_1_batch_normalization_betaC
?fusedbatchnormv3_readvariableop_batch_normalization_moving_meanI
Efusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance
identityИЕ
ReadVariableOpReadVariableOp(readvariableop_batch_normalization_gamma*
_output_shapes
: *
dtype02
ReadVariableOpК
ReadVariableOp_1ReadVariableOp)readvariableop_1_batch_normalization_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1╛
FusedBatchNormV3/ReadVariableOpReadVariableOp?fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp╚
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpEfusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1┴
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%oГ:*
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
Л
З
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2225

inputs.
*readvariableop_batch_normalization_6_gamma/
+readvariableop_1_batch_normalization_6_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_6_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance
identityИвAssignNewValueвAssignNewValue_1И
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_6_gamma*
_output_shapes	
:А*
dtype02
ReadVariableOpН
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_6_beta*
_output_shapes	
:А*
dtype02
ReadVariableOp_1┴
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_6_moving_mean*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOp╦
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╘
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3:А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3▒
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_6_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_6/moving_mean*
_output_shapes
 *
dtype02
AssignNewValue╟
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_6/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1М
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*'
_output_shapes
:А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:А::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:O K
'
_output_shapes
:А
 
_user_specified_nameinputs
╤
¤
M__inference_batch_normalization_layer_call_and_return_conditional_losses_3917

inputs,
(readvariableop_batch_normalization_gamma-
)readvariableop_1_batch_normalization_betaC
?fusedbatchnormv3_readvariableop_batch_normalization_moving_meanI
Efusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance
identityИвAssignNewValueвAssignNewValue_1Е
ReadVariableOpReadVariableOp(readvariableop_batch_normalization_gamma*
_output_shapes
: *
dtype02
ReadVariableOpК
ReadVariableOp_1ReadVariableOp)readvariableop_1_batch_normalization_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1╛
FusedBatchNormV3/ReadVariableOpReadVariableOp?fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp╚
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpEfusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3н
AssignNewValueAssignVariableOp?fusedbatchnormv3_readvariableop_batch_normalization_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization/moving_mean*
_output_shapes
 *
dtype02
AssignNewValue├
AssignNewValue_1AssignVariableOpEfusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*X
_classN
LJloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1ж
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
п
]
A__inference_re_lu_4_layer_call_and_return_conditional_losses_2103

inputs
identityP
Relu6Relu6inputs*
T0*&
_output_shapes
:@2
Relu6f
IdentityIdentityRelu6:activations:0*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*%
_input_shapes
:@:N J
&
_output_shapes
:@
 
_user_specified_nameinputs
╝
q
%__inference_conv2d_layer_call_fn_3899

inputs
conv2d_kernel
identityИвStatefulPartitionedCallш
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
GPU 2J 8В *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_16902
StatefulPartitionedCallН
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
╕	
Ў
2__inference_batch_normalization_layer_call_fn_3953

inputs
batch_normalization_gamma
batch_normalization_beta#
batch_normalization_moving_mean'
#batch_normalization_moving_variance
identityИвStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_gammabatch_normalization_betabatch_normalization_moving_mean#batch_normalization_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_batch_normalization_layer_call_and_return_conditional_losses_7992
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
╧
у
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4107

inputs.
*readvariableop_batch_normalization_1_gamma/
+readvariableop_1_batch_normalization_1_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_1_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance
identityИЗ
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_1_gamma*
_output_shapes
: *
dtype02
ReadVariableOpМ
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_1_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1└
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp╩
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            :::::i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
ъ
Ж
N__inference_batch_normalization_2_layer_call_and_return_conditional_losses_982

inputs.
*readvariableop_batch_normalization_2_gamma/
+readvariableop_1_batch_normalization_2_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_2_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance
identityИвAssignNewValueвAssignNewValue_1З
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_2_gamma*
_output_shapes
: *
dtype02
ReadVariableOpМ
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_2_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1└
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp╩
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3▒
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_2_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_2/moving_mean*
_output_shapes
 *
dtype02
AssignNewValue╟
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_2/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1ж
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
╩	
А
4__inference_batch_normalization_2_layer_call_fn_4247

inputs
batch_normalization_2_gamma
batch_normalization_2_beta%
!batch_normalization_2_moving_mean)
%batch_normalization_2_moving_variance
identityИвStatefulPartitionedCallЙ
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_2_gammabatch_normalization_2_beta!batch_normalization_2_moving_mean%batch_normalization_2_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_batch_normalization_2_layer_call_and_return_conditional_losses_9822
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
╔
М
B__inference_conv2d_1_layer_call_and_return_conditional_losses_4142

inputs)
%conv2d_readvariableop_conv2d_1_kernel
identityИЬ
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_1_kernel*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOpЪ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingSAME*
strides
2
Conv2Db
IdentityIdentityConv2D:output:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*)
_input_shapes
: ::N J
&
_output_shapes
: 
 
_user_specified_nameinputs
╝

$__inference_dense_layer_call_fn_5057

inputs
dense_kernel

dense_bias
identityИвStatefulPartitionedCallь
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
GPU 2J 8В *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_24952
StatefulPartitionedCallЕ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*&
_input_shapes
:	А::22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	А
 
_user_specified_nameinputs
ы
З
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_4587

inputs.
*readvariableop_batch_normalization_5_gamma/
+readvariableop_1_batch_normalization_5_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_5_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance
identityИвAssignNewValueвAssignNewValue_1З
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_5_gamma*
_output_shapes
:@*
dtype02
ReadVariableOpМ
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_5_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1└
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_5_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp╩
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3▒
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_5_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_5/moving_mean*
_output_shapes
 *
dtype02
AssignNewValue╟
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_5/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1ж
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
ы
З
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4089

inputs.
*readvariableop_batch_normalization_1_gamma/
+readvariableop_1_batch_normalization_1_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_1_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance
identityИвAssignNewValueвAssignNewValue_1З
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_1_gamma*
_output_shapes
: *
dtype02
ReadVariableOpМ
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_1_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1└
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp╩
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3▒
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_1_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_1/moving_mean*
_output_shapes
 *
dtype02
AssignNewValue╟
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_1/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1ж
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
 
З
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_4415

inputs.
*readvariableop_batch_normalization_4_gamma/
+readvariableop_1_batch_normalization_4_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_4_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance
identityИвAssignNewValueвAssignNewValue_1З
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_4_gamma*
_output_shapes
:@*
dtype02
ReadVariableOpМ
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_4_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1└
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp╩
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╧
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3▒
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_4_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_4/moving_mean*
_output_shapes
 *
dtype02
AssignNewValue╟
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_4/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1Л
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*5
_input_shapes$
":@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:N J
&
_output_shapes
:@
 
_user_specified_nameinputs
ы
З
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1310

inputs.
*readvariableop_batch_normalization_5_gamma/
+readvariableop_1_batch_normalization_5_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_5_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance
identityИвAssignNewValueвAssignNewValue_1З
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_5_gamma*
_output_shapes
:@*
dtype02
ReadVariableOpМ
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_5_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1└
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_5_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp╩
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3▒
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_5_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_5/moving_mean*
_output_shapes
 *
dtype02
AssignNewValue╟
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_5/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1ж
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
ў
З
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_4664

inputs.
*readvariableop_batch_normalization_6_gamma/
+readvariableop_1_batch_normalization_6_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_6_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance
identityИвAssignNewValueвAssignNewValue_1И
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_6_gamma*
_output_shapes	
:А*
dtype02
ReadVariableOpН
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_6_beta*
_output_shapes	
:А*
dtype02
ReadVariableOp_1┴
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_6_moving_mean*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOp╦
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1я
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3▒
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_6_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_6/moving_mean*
_output_shapes
 *
dtype02
AssignNewValue╟
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_6/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1з
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,                           А::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
хв
Б#
F__inference_functional_1_layer_call_and_return_conditional_losses_3586

inputs.
*conv2d_conv2d_readvariableop_conv2d_kernel@
<batch_normalization_readvariableop_batch_normalization_gammaA
=batch_normalization_readvariableop_1_batch_normalization_betaW
Sbatch_normalization_fusedbatchnormv3_readvariableop_batch_normalization_moving_mean]
Ybatch_normalization_fusedbatchnormv3_readvariableop_1_batch_normalization_moving_varianceO
Kdepthwise_conv2d_depthwise_readvariableop_depthwise_conv2d_depthwise_kernelD
@batch_normalization_1_readvariableop_batch_normalization_1_gammaE
Abatch_normalization_1_readvariableop_1_batch_normalization_1_beta[
Wbatch_normalization_1_fusedbatchnormv3_readvariableop_batch_normalization_1_moving_meana
]batch_normalization_1_fusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance2
.conv2d_1_conv2d_readvariableop_conv2d_1_kernelD
@batch_normalization_2_readvariableop_batch_normalization_2_gammaE
Abatch_normalization_2_readvariableop_1_batch_normalization_2_beta[
Wbatch_normalization_2_fusedbatchnormv3_readvariableop_batch_normalization_2_moving_meana
]batch_normalization_2_fusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_varianceS
Odepthwise_conv2d_1_depthwise_readvariableop_depthwise_conv2d_1_depthwise_kernelD
@batch_normalization_3_readvariableop_batch_normalization_3_gammaE
Abatch_normalization_3_readvariableop_1_batch_normalization_3_beta[
Wbatch_normalization_3_fusedbatchnormv3_readvariableop_batch_normalization_3_moving_meana
]batch_normalization_3_fusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance2
.conv2d_2_conv2d_readvariableop_conv2d_2_kernelD
@batch_normalization_4_readvariableop_batch_normalization_4_gammaE
Abatch_normalization_4_readvariableop_1_batch_normalization_4_beta[
Wbatch_normalization_4_fusedbatchnormv3_readvariableop_batch_normalization_4_moving_meana
]batch_normalization_4_fusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_varianceS
Odepthwise_conv2d_2_depthwise_readvariableop_depthwise_conv2d_2_depthwise_kernelD
@batch_normalization_5_readvariableop_batch_normalization_5_gammaE
Abatch_normalization_5_readvariableop_1_batch_normalization_5_beta[
Wbatch_normalization_5_fusedbatchnormv3_readvariableop_batch_normalization_5_moving_meana
]batch_normalization_5_fusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance2
.conv2d_3_conv2d_readvariableop_conv2d_3_kernelD
@batch_normalization_6_readvariableop_batch_normalization_6_gammaE
Abatch_normalization_6_readvariableop_1_batch_normalization_6_beta[
Wbatch_normalization_6_fusedbatchnormv3_readvariableop_batch_normalization_6_moving_meana
]batch_normalization_6_fusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_varianceS
Odepthwise_conv2d_3_depthwise_readvariableop_depthwise_conv2d_3_depthwise_kernelD
@batch_normalization_7_readvariableop_batch_normalization_7_gammaE
Abatch_normalization_7_readvariableop_1_batch_normalization_7_beta[
Wbatch_normalization_7_fusedbatchnormv3_readvariableop_batch_normalization_7_moving_meana
]batch_normalization_7_fusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance2
.conv2d_4_conv2d_readvariableop_conv2d_4_kernelD
@batch_normalization_8_readvariableop_batch_normalization_8_gammaE
Abatch_normalization_8_readvariableop_1_batch_normalization_8_beta[
Wbatch_normalization_8_fusedbatchnormv3_readvariableop_batch_normalization_8_moving_meana
]batch_normalization_8_fusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance,
(dense_matmul_readvariableop_dense_kernel+
'dense_biasadd_readvariableop_dense_bias
identityИв"batch_normalization/AssignNewValueв$batch_normalization/AssignNewValue_1в$batch_normalization_1/AssignNewValueв&batch_normalization_1/AssignNewValue_1в$batch_normalization_2/AssignNewValueв&batch_normalization_2/AssignNewValue_1в$batch_normalization_3/AssignNewValueв&batch_normalization_3/AssignNewValue_1в$batch_normalization_4/AssignNewValueв&batch_normalization_4/AssignNewValue_1в$batch_normalization_5/AssignNewValueв&batch_normalization_5/AssignNewValue_1в$batch_normalization_6/AssignNewValueв&batch_normalization_6/AssignNewValue_1в$batch_normalization_7/AssignNewValueв&batch_normalization_7/AssignNewValue_1в$batch_normalization_8/AssignNewValueв&batch_normalization_8/AssignNewValue_1Р
%tf_op_layer_ExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%tf_op_layer_ExpandDims/ExpandDims/dim╠
!tf_op_layer_ExpandDims/ExpandDims
ExpandDimsinputs.tf_op_layer_ExpandDims/ExpandDims/dim:output:0*
T0*
_cloned(*&
_output_shapes
:1(2#
!tf_op_layer_ExpandDims/ExpandDimsп
conv2d/Conv2D/ReadVariableOpReadVariableOp*conv2d_conv2d_readvariableop_conv2d_kernel*&
_output_shapes
:( *
dtype02
conv2d/Conv2D/ReadVariableOp╘
conv2d/Conv2DConv2D*tf_op_layer_ExpandDims/ExpandDims:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
2
conv2d/Conv2D┴
"batch_normalization/ReadVariableOpReadVariableOp<batch_normalization_readvariableop_batch_normalization_gamma*
_output_shapes
: *
dtype02$
"batch_normalization/ReadVariableOp╞
$batch_normalization/ReadVariableOp_1ReadVariableOp=batch_normalization_readvariableop_1_batch_normalization_beta*
_output_shapes
: *
dtype02&
$batch_normalization/ReadVariableOp_1·
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpSbatch_normalization_fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
_output_shapes
: *
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpД
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpYbatch_normalization_fusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
_output_shapes
: *
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1╫
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/Conv2D:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2&
$batch_normalization/FusedBatchNormV3е
"batch_normalization/AssignNewValueAssignVariableOpSbatch_normalization_fusedbatchnormv3_readvariableop_batch_normalization_moving_mean1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*f
_class\
ZXloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp/batch_normalization/moving_mean*
_output_shapes
 *
dtype02$
"batch_normalization/AssignNewValue╗
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
re_lu/Relu6ъ
)depthwise_conv2d/depthwise/ReadVariableOpReadVariableOpKdepthwise_conv2d_depthwise_readvariableop_depthwise_conv2d_depthwise_kernel*&
_output_shapes
: *
dtype02+
)depthwise_conv2d/depthwise/ReadVariableOpЭ
 depthwise_conv2d/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 depthwise_conv2d/depthwise/Shapeе
(depthwise_conv2d/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2*
(depthwise_conv2d/depthwise/dilation_rate∙
depthwise_conv2d/depthwiseDepthwiseConv2dNativere_lu/Relu6:activations:01depthwise_conv2d/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
2
depthwise_conv2d/depthwise╔
$batch_normalization_1/ReadVariableOpReadVariableOp@batch_normalization_1_readvariableop_batch_normalization_1_gamma*
_output_shapes
: *
dtype02&
$batch_normalization_1/ReadVariableOp╬
&batch_normalization_1/ReadVariableOp_1ReadVariableOpAbatch_normalization_1_readvariableop_1_batch_normalization_1_beta*
_output_shapes
: *
dtype02(
&batch_normalization_1/ReadVariableOp_1В
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_1_fusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
: *
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpМ
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_1_fusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
_output_shapes
: *
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Ё
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3#depthwise_conv2d/depthwise:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2(
&batch_normalization_1/FusedBatchNormV3╡
$batch_normalization_1/AssignNewValueAssignVariableOpWbatch_normalization_1_fusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*j
_class`
^\loc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp/batch_normalization_1/moving_mean*
_output_shapes
 *
dtype02&
$batch_normalization_1/AssignNewValue╦
&batch_normalization_1/AssignNewValue_1AssignVariableOp]batch_normalization_1_fusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*p
_classf
dbloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_1/moving_variance*
_output_shapes
 *
dtype02(
&batch_normalization_1/AssignNewValue_1Д
re_lu_1/Relu6Relu6*batch_normalization_1/FusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2
re_lu_1/Relu6╖
conv2d_1/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_conv2d_readvariableop_conv2d_1_kernel*&
_output_shapes
:  *
dtype02 
conv2d_1/Conv2D/ReadVariableOp╩
conv2d_1/Conv2DConv2Dre_lu_1/Relu6:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingSAME*
strides
2
conv2d_1/Conv2D╔
$batch_normalization_2/ReadVariableOpReadVariableOp@batch_normalization_2_readvariableop_batch_normalization_2_gamma*
_output_shapes
: *
dtype02&
$batch_normalization_2/ReadVariableOp╬
&batch_normalization_2/ReadVariableOp_1ReadVariableOpAbatch_normalization_2_readvariableop_1_batch_normalization_2_beta*
_output_shapes
: *
dtype02(
&batch_normalization_2/ReadVariableOp_1В
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_2_fusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
: *
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpМ
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_2_fusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
_output_shapes
: *
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1х
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_1/Conv2D:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2(
&batch_normalization_2/FusedBatchNormV3╡
$batch_normalization_2/AssignNewValueAssignVariableOpWbatch_normalization_2_fusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*j
_class`
^\loc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp/batch_normalization_2/moving_mean*
_output_shapes
 *
dtype02&
$batch_normalization_2/AssignNewValue╦
&batch_normalization_2/AssignNewValue_1AssignVariableOp]batch_normalization_2_fusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*p
_classf
dbloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_2/moving_variance*
_output_shapes
 *
dtype02(
&batch_normalization_2/AssignNewValue_1Д
re_lu_2/Relu6Relu6*batch_normalization_2/FusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2
re_lu_2/Relu6Є
+depthwise_conv2d_1/depthwise/ReadVariableOpReadVariableOpOdepthwise_conv2d_1_depthwise_readvariableop_depthwise_conv2d_1_depthwise_kernel*&
_output_shapes
: *
dtype02-
+depthwise_conv2d_1/depthwise/ReadVariableOpб
"depthwise_conv2d_1/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"depthwise_conv2d_1/depthwise/Shapeй
*depthwise_conv2d_1/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2,
*depthwise_conv2d_1/depthwise/dilation_rateБ
depthwise_conv2d_1/depthwiseDepthwiseConv2dNativere_lu_2/Relu6:activations:03depthwise_conv2d_1/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
2
depthwise_conv2d_1/depthwise╔
$batch_normalization_3/ReadVariableOpReadVariableOp@batch_normalization_3_readvariableop_batch_normalization_3_gamma*
_output_shapes
: *
dtype02&
$batch_normalization_3/ReadVariableOp╬
&batch_normalization_3/ReadVariableOp_1ReadVariableOpAbatch_normalization_3_readvariableop_1_batch_normalization_3_beta*
_output_shapes
: *
dtype02(
&batch_normalization_3/ReadVariableOp_1В
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_3_fusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
_output_shapes
: *
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpМ
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_3_fusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
_output_shapes
: *
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Є
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3%depthwise_conv2d_1/depthwise:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2(
&batch_normalization_3/FusedBatchNormV3╡
$batch_normalization_3/AssignNewValueAssignVariableOpWbatch_normalization_3_fusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*j
_class`
^\loc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp/batch_normalization_3/moving_mean*
_output_shapes
 *
dtype02&
$batch_normalization_3/AssignNewValue╦
&batch_normalization_3/AssignNewValue_1AssignVariableOp]batch_normalization_3_fusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance7batch_normalization_3/FusedBatchNormV3:batch_variance:08^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*p
_classf
dbloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_3/moving_variance*
_output_shapes
 *
dtype02(
&batch_normalization_3/AssignNewValue_1Д
re_lu_3/Relu6Relu6*batch_normalization_3/FusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2
re_lu_3/Relu6╖
conv2d_2/Conv2D/ReadVariableOpReadVariableOp.conv2d_2_conv2d_readvariableop_conv2d_2_kernel*&
_output_shapes
: @*
dtype02 
conv2d_2/Conv2D/ReadVariableOp╩
conv2d_2/Conv2DConv2Dre_lu_3/Relu6:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingSAME*
strides
2
conv2d_2/Conv2D╔
$batch_normalization_4/ReadVariableOpReadVariableOp@batch_normalization_4_readvariableop_batch_normalization_4_gamma*
_output_shapes
:@*
dtype02&
$batch_normalization_4/ReadVariableOp╬
&batch_normalization_4/ReadVariableOp_1ReadVariableOpAbatch_normalization_4_readvariableop_1_batch_normalization_4_beta*
_output_shapes
:@*
dtype02(
&batch_normalization_4/ReadVariableOp_1В
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_4_fusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
_output_shapes
:@*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpМ
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_4_fusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
_output_shapes
:@*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1х
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_2/Conv2D:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2(
&batch_normalization_4/FusedBatchNormV3╡
$batch_normalization_4/AssignNewValueAssignVariableOpWbatch_normalization_4_fusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean3batch_normalization_4/FusedBatchNormV3:batch_mean:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp*j
_class`
^\loc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp/batch_normalization_4/moving_mean*
_output_shapes
 *
dtype02&
$batch_normalization_4/AssignNewValue╦
&batch_normalization_4/AssignNewValue_1AssignVariableOp]batch_normalization_4_fusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance7batch_normalization_4/FusedBatchNormV3:batch_variance:08^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*p
_classf
dbloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_4/moving_variance*
_output_shapes
 *
dtype02(
&batch_normalization_4/AssignNewValue_1Д
re_lu_4/Relu6Relu6*batch_normalization_4/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:@2
re_lu_4/Relu6Є
+depthwise_conv2d_2/depthwise/ReadVariableOpReadVariableOpOdepthwise_conv2d_2_depthwise_readvariableop_depthwise_conv2d_2_depthwise_kernel*&
_output_shapes
:@*
dtype02-
+depthwise_conv2d_2/depthwise/ReadVariableOpб
"depthwise_conv2d_2/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @      2$
"depthwise_conv2d_2/depthwise/Shapeй
*depthwise_conv2d_2/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2,
*depthwise_conv2d_2/depthwise/dilation_rateА
depthwise_conv2d_2/depthwiseDepthwiseConv2dNativere_lu_4/Relu6:activations:03depthwise_conv2d_2/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingSAME*
strides
2
depthwise_conv2d_2/depthwise╔
$batch_normalization_5/ReadVariableOpReadVariableOp@batch_normalization_5_readvariableop_batch_normalization_5_gamma*
_output_shapes
:@*
dtype02&
$batch_normalization_5/ReadVariableOp╬
&batch_normalization_5/ReadVariableOp_1ReadVariableOpAbatch_normalization_5_readvariableop_1_batch_normalization_5_beta*
_output_shapes
:@*
dtype02(
&batch_normalization_5/ReadVariableOp_1В
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_5_fusedbatchnormv3_readvariableop_batch_normalization_5_moving_mean*
_output_shapes
:@*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpМ
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_5_fusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance*
_output_shapes
:@*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Є
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3%depthwise_conv2d_2/depthwise:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2(
&batch_normalization_5/FusedBatchNormV3╡
$batch_normalization_5/AssignNewValueAssignVariableOpWbatch_normalization_5_fusedbatchnormv3_readvariableop_batch_normalization_5_moving_mean3batch_normalization_5/FusedBatchNormV3:batch_mean:06^batch_normalization_5/FusedBatchNormV3/ReadVariableOp*j
_class`
^\loc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp/batch_normalization_5/moving_mean*
_output_shapes
 *
dtype02&
$batch_normalization_5/AssignNewValue╦
&batch_normalization_5/AssignNewValue_1AssignVariableOp]batch_normalization_5_fusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance7batch_normalization_5/FusedBatchNormV3:batch_variance:08^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*p
_classf
dbloc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_5/moving_variance*
_output_shapes
 *
dtype02(
&batch_normalization_5/AssignNewValue_1Д
re_lu_5/Relu6Relu6*batch_normalization_5/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:@2
re_lu_5/Relu6╕
conv2d_3/Conv2D/ReadVariableOpReadVariableOp.conv2d_3_conv2d_readvariableop_conv2d_3_kernel*'
_output_shapes
:@А*
dtype02 
conv2d_3/Conv2D/ReadVariableOp╦
conv2d_3/Conv2DConv2Dre_lu_5/Relu6:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:А*
paddingSAME*
strides
2
conv2d_3/Conv2D╩
$batch_normalization_6/ReadVariableOpReadVariableOp@batch_normalization_6_readvariableop_batch_normalization_6_gamma*
_output_shapes	
:А*
dtype02&
$batch_normalization_6/ReadVariableOp╧
&batch_normalization_6/ReadVariableOp_1ReadVariableOpAbatch_normalization_6_readvariableop_1_batch_normalization_6_beta*
_output_shapes	
:А*
dtype02(
&batch_normalization_6/ReadVariableOp_1Г
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_6_fusedbatchnormv3_readvariableop_batch_normalization_6_moving_mean*
_output_shapes	
:А*
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpН
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_6_fusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance*
_output_shapes	
:А*
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ъ
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv2d_3/Conv2D:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3:А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2(
&batch_normalization_6/FusedBatchNormV3╡
$batch_normalization_6/AssignNewValueAssignVariableOpWbatch_normalization_6_fusedbatchnormv3_readvariableop_batch_normalization_6_moving_mean3batch_normalization_6/FusedBatchNormV3:batch_mean:06^batch_normalization_6/FusedBatchNormV3/ReadVariableOp*j
_class`
^\loc:@batch_normalization_6/FusedBatchNormV3/ReadVariableOp/batch_normalization_6/moving_mean*
_output_shapes
 *
dtype02&
$batch_normalization_6/AssignNewValue╦
&batch_normalization_6/AssignNewValue_1AssignVariableOp]batch_normalization_6_fusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance7batch_normalization_6/FusedBatchNormV3:batch_variance:08^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1*p
_classf
dbloc:@batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_6/moving_variance*
_output_shapes
 *
dtype02(
&batch_normalization_6/AssignNewValue_1Е
re_lu_6/Relu6Relu6*batch_normalization_6/FusedBatchNormV3:y:0*
T0*'
_output_shapes
:А2
re_lu_6/Relu6є
+depthwise_conv2d_3/depthwise/ReadVariableOpReadVariableOpOdepthwise_conv2d_3_depthwise_readvariableop_depthwise_conv2d_3_depthwise_kernel*'
_output_shapes
:А*
dtype02-
+depthwise_conv2d_3/depthwise/ReadVariableOpб
"depthwise_conv2d_3/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      А      2$
"depthwise_conv2d_3/depthwise/Shapeй
*depthwise_conv2d_3/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2,
*depthwise_conv2d_3/depthwise/dilation_rateБ
depthwise_conv2d_3/depthwiseDepthwiseConv2dNativere_lu_6/Relu6:activations:03depthwise_conv2d_3/depthwise/ReadVariableOp:value:0*
T0*'
_output_shapes
:А*
paddingSAME*
strides
2
depthwise_conv2d_3/depthwise╩
$batch_normalization_7/ReadVariableOpReadVariableOp@batch_normalization_7_readvariableop_batch_normalization_7_gamma*
_output_shapes	
:А*
dtype02&
$batch_normalization_7/ReadVariableOp╧
&batch_normalization_7/ReadVariableOp_1ReadVariableOpAbatch_normalization_7_readvariableop_1_batch_normalization_7_beta*
_output_shapes	
:А*
dtype02(
&batch_normalization_7/ReadVariableOp_1Г
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_7_fusedbatchnormv3_readvariableop_batch_normalization_7_moving_mean*
_output_shapes	
:А*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpН
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_7_fusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance*
_output_shapes	
:А*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ў
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3%depthwise_conv2d_3/depthwise:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3:А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2(
&batch_normalization_7/FusedBatchNormV3╡
$batch_normalization_7/AssignNewValueAssignVariableOpWbatch_normalization_7_fusedbatchnormv3_readvariableop_batch_normalization_7_moving_mean3batch_normalization_7/FusedBatchNormV3:batch_mean:06^batch_normalization_7/FusedBatchNormV3/ReadVariableOp*j
_class`
^\loc:@batch_normalization_7/FusedBatchNormV3/ReadVariableOp/batch_normalization_7/moving_mean*
_output_shapes
 *
dtype02&
$batch_normalization_7/AssignNewValue╦
&batch_normalization_7/AssignNewValue_1AssignVariableOp]batch_normalization_7_fusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance7batch_normalization_7/FusedBatchNormV3:batch_variance:08^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1*p
_classf
dbloc:@batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_7/moving_variance*
_output_shapes
 *
dtype02(
&batch_normalization_7/AssignNewValue_1Е
re_lu_7/Relu6Relu6*batch_normalization_7/FusedBatchNormV3:y:0*
T0*'
_output_shapes
:А2
re_lu_7/Relu6╣
conv2d_4/Conv2D/ReadVariableOpReadVariableOp.conv2d_4_conv2d_readvariableop_conv2d_4_kernel*(
_output_shapes
:АА*
dtype02 
conv2d_4/Conv2D/ReadVariableOp╦
conv2d_4/Conv2DConv2Dre_lu_7/Relu6:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:А*
paddingSAME*
strides
2
conv2d_4/Conv2D╩
$batch_normalization_8/ReadVariableOpReadVariableOp@batch_normalization_8_readvariableop_batch_normalization_8_gamma*
_output_shapes	
:А*
dtype02&
$batch_normalization_8/ReadVariableOp╧
&batch_normalization_8/ReadVariableOp_1ReadVariableOpAbatch_normalization_8_readvariableop_1_batch_normalization_8_beta*
_output_shapes	
:А*
dtype02(
&batch_normalization_8/ReadVariableOp_1Г
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_8_fusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean*
_output_shapes	
:А*
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpН
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_8_fusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance*
_output_shapes	
:А*
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ъ
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3conv2d_4/Conv2D:output:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3:А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2(
&batch_normalization_8/FusedBatchNormV3╡
$batch_normalization_8/AssignNewValueAssignVariableOpWbatch_normalization_8_fusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean3batch_normalization_8/FusedBatchNormV3:batch_mean:06^batch_normalization_8/FusedBatchNormV3/ReadVariableOp*j
_class`
^\loc:@batch_normalization_8/FusedBatchNormV3/ReadVariableOp/batch_normalization_8/moving_mean*
_output_shapes
 *
dtype02&
$batch_normalization_8/AssignNewValue╦
&batch_normalization_8/AssignNewValue_1AssignVariableOp]batch_normalization_8_fusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance7batch_normalization_8/FusedBatchNormV3:batch_variance:08^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1*p
_classf
dbloc:@batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_8/moving_variance*
_output_shapes
 *
dtype02(
&batch_normalization_8/AssignNewValue_1Е
re_lu_8/Relu6Relu6*batch_normalization_8/FusedBatchNormV3:y:0*
T0*'
_output_shapes
:А2
re_lu_8/Relu6│
/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      21
/global_average_pooling2d/Mean/reduction_indices╟
global_average_pooling2d/MeanMeanre_lu_8/Relu6:activations:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*
_output_shapes
:	А2
global_average_pooling2d/Means
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/dropout/Constг
dropout/dropout/MulMul&global_average_pooling2d/Mean:output:0dropout/dropout/Const:output:0*
T0*
_output_shapes
:	А2
dropout/dropout/Mul
dropout/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   А   2
dropout/dropout/Shape─
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*
_output_shapes
:	А*
dtype02.
,dropout/dropout/random_uniform/RandomUniformЕ
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2 
dropout/dropout/GreaterEqual/y╓
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	А2
dropout/dropout/GreaterEqualП
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	А2
dropout/dropout/CastТ
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*
_output_shapes
:	А2
dropout/dropout/Mul_1д
dense/MatMul/ReadVariableOpReadVariableOp(dense_matmul_readvariableop_dense_kernel*
_output_shapes
:	А*
dtype02
dense/MatMul/ReadVariableOpП
dense/MatMulMatMuldropout/dropout/Mul_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense/MatMulа
dense/BiasAdd/ReadVariableOpReadVariableOp'dense_biasadd_readvariableop_dense_bias*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOpР
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense/BiasAddн
IdentityIdentitydense/BiasAdd:output:0#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_1%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_1%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_1%^batch_normalization_4/AssignNewValue'^batch_normalization_4/AssignNewValue_1%^batch_normalization_5/AssignNewValue'^batch_normalization_5/AssignNewValue_1%^batch_normalization_6/AssignNewValue'^batch_normalization_6/AssignNewValue_1%^batch_normalization_7/AssignNewValue'^batch_normalization_7/AssignNewValue_1%^batch_normalization_8/AssignNewValue'^batch_normalization_8/AssignNewValue_1*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*▀
_input_shapes═
╩:1(:::::::::::::::::::::::::::::::::::::::::::::::2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12L
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
&batch_normalization_8/AssignNewValue_1&batch_normalization_8/AssignNewValue_1:S O
+
_output_shapes
:         1(
 
_user_specified_nameinputs
с
А
4__inference_batch_normalization_2_layer_call_fn_4202

inputs
batch_normalization_2_gamma
batch_normalization_2_beta%
!batch_normalization_2_moving_mean)
%batch_normalization_2_moving_variance
identityИвStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_2_gammabatch_normalization_2_beta!batch_normalization_2_moving_mean%batch_normalization_2_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_19052
StatefulPartitionedCallН
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": ::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
: 
 
_user_specified_nameinputs
ё
У
1__inference_depthwise_conv2d_3_layer_call_fn_1462

inputs'
#depthwise_conv2d_3_depthwise_kernel
identityИвStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallinputs#depthwise_conv2d_3_depthwise_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_depthwise_conv2d_3_layer_call_and_return_conditional_losses_14582
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:,                           А:22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
─
u
'__inference_conv2d_2_layer_call_fn_4397

inputs
conv2d_2_kernel
identityИвStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_2_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_20292
StatefulPartitionedCallН
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*)
_input_shapes
: :22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
: 
 
_user_specified_nameinputs
█
у
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_4682

inputs.
*readvariableop_batch_normalization_6_gamma/
+readvariableop_1_batch_normalization_6_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_6_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance
identityИИ
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_6_gamma*
_output_shapes	
:А*
dtype02
ReadVariableOpН
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_6_beta*
_output_shapes	
:А*
dtype02
ReadVariableOp_1┴
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_6_moving_mean*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOp╦
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1с
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Г
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,                           А:::::j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
╧
у
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1009

inputs.
*readvariableop_batch_normalization_2_gamma/
+readvariableop_1_batch_normalization_2_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_2_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance
identityИЗ
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_2_gamma*
_output_shapes
: *
dtype02
ReadVariableOpМ
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_2_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1└
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp╩
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            :::::i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
╔
М
B__inference_conv2d_1_layer_call_and_return_conditional_losses_1860

inputs)
%conv2d_readvariableop_conv2d_1_kernel
identityИЬ
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_1_kernel*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOpЪ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingSAME*
strides
2
Conv2Db
IdentityIdentityConv2D:output:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*)
_input_shapes
: ::N J
&
_output_shapes
: 
 
_user_specified_nameinputs
т
у
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1905

inputs.
*readvariableop_batch_normalization_2_gamma/
+readvariableop_1_batch_normalization_2_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_2_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance
identityИЗ
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_2_gamma*
_output_shapes
: *
dtype02
ReadVariableOpМ
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_2_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1└
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp╩
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1┴
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3g
IdentityIdentityFusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": :::::N J
&
_output_shapes
: 
 
_user_specified_nameinputs
п
]
A__inference_re_lu_2_layer_call_and_return_conditional_losses_1934

inputs
identityP
Relu6Relu6inputs*
T0*&
_output_shapes
: 2
Relu6f
IdentityIdentityRelu6:activations:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*%
_input_shapes
: :N J
&
_output_shapes
: 
 
_user_specified_nameinputs
Й 
│
+__inference_functional_1_layer_call_fn_3342
input_1
conv2d_kernel
batch_normalization_gamma
batch_normalization_beta#
batch_normalization_moving_mean'
#batch_normalization_moving_variance%
!depthwise_conv2d_depthwise_kernel
batch_normalization_1_gamma
batch_normalization_1_beta%
!batch_normalization_1_moving_mean)
%batch_normalization_1_moving_variance
conv2d_1_kernel
batch_normalization_2_gamma
batch_normalization_2_beta%
!batch_normalization_2_moving_mean)
%batch_normalization_2_moving_variance'
#depthwise_conv2d_1_depthwise_kernel
batch_normalization_3_gamma
batch_normalization_3_beta%
!batch_normalization_3_moving_mean)
%batch_normalization_3_moving_variance
conv2d_2_kernel
batch_normalization_4_gamma
batch_normalization_4_beta%
!batch_normalization_4_moving_mean)
%batch_normalization_4_moving_variance'
#depthwise_conv2d_2_depthwise_kernel
batch_normalization_5_gamma
batch_normalization_5_beta%
!batch_normalization_5_moving_mean)
%batch_normalization_5_moving_variance
conv2d_3_kernel
batch_normalization_6_gamma
batch_normalization_6_beta%
!batch_normalization_6_moving_mean)
%batch_normalization_6_moving_variance'
#depthwise_conv2d_3_depthwise_kernel
batch_normalization_7_gamma
batch_normalization_7_beta%
!batch_normalization_7_moving_mean)
%batch_normalization_7_moving_variance
conv2d_4_kernel
batch_normalization_8_gamma
batch_normalization_8_beta%
!batch_normalization_8_moving_mean)
%batch_normalization_8_moving_variance
dense_kernel

dense_bias
identityИвStatefulPartitionedCall┤
StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_kernelbatch_normalization_gammabatch_normalization_betabatch_normalization_moving_mean#batch_normalization_moving_variance!depthwise_conv2d_depthwise_kernelbatch_normalization_1_gammabatch_normalization_1_beta!batch_normalization_1_moving_mean%batch_normalization_1_moving_varianceconv2d_1_kernelbatch_normalization_2_gammabatch_normalization_2_beta!batch_normalization_2_moving_mean%batch_normalization_2_moving_variance#depthwise_conv2d_1_depthwise_kernelbatch_normalization_3_gammabatch_normalization_3_beta!batch_normalization_3_moving_mean%batch_normalization_3_moving_varianceconv2d_2_kernelbatch_normalization_4_gammabatch_normalization_4_beta!batch_normalization_4_moving_mean%batch_normalization_4_moving_variance#depthwise_conv2d_2_depthwise_kernelbatch_normalization_5_gammabatch_normalization_5_beta!batch_normalization_5_moving_mean%batch_normalization_5_moving_varianceconv2d_3_kernelbatch_normalization_6_gammabatch_normalization_6_beta!batch_normalization_6_moving_mean%batch_normalization_6_moving_variance#depthwise_conv2d_3_depthwise_kernelbatch_normalization_7_gammabatch_normalization_7_beta!batch_normalization_7_moving_mean%batch_normalization_7_moving_varianceconv2d_4_kernelbatch_normalization_8_gammabatch_normalization_8_beta!batch_normalization_8_moving_mean%batch_normalization_8_moving_variancedense_kernel
dense_bias*;
Tin4
220*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*?
_read_only_resource_inputs!
 !$%&)*+./*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_functional_1_layer_call_and_return_conditional_losses_26752
StatefulPartitionedCallЕ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*ш
_input_shapes╓
╙:         1(:::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:         1(
!
_user_specified_name	input_1
Л
З
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2394

inputs.
*readvariableop_batch_normalization_8_gamma/
+readvariableop_1_batch_normalization_8_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_8_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance
identityИвAssignNewValueвAssignNewValue_1И
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_8_gamma*
_output_shapes	
:А*
dtype02
ReadVariableOpН
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_8_beta*
_output_shapes	
:А*
dtype02
ReadVariableOp_1┴
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOp╦
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╘
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3:А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3▒
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_8_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_8/moving_mean*
_output_shapes
 *
dtype02
AssignNewValue╟
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_8/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1М
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*'
_output_shapes
:А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:А::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:O K
'
_output_shapes
:А
 
_user_specified_nameinputs
═	
А
4__inference_batch_normalization_2_layer_call_fn_4256

inputs
batch_normalization_2_gamma
batch_normalization_2_beta%
!batch_normalization_2_moving_mean)
%batch_normalization_2_moving_variance
identityИвStatefulPartitionedCallМ
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_2_gammabatch_normalization_2_beta!batch_normalization_2_moving_mean%batch_normalization_2_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_10092
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
╦
Ў
2__inference_batch_normalization_layer_call_fn_3998

inputs
batch_normalization_gamma
batch_normalization_beta#
batch_normalization_moving_mean'
#batch_normalization_moving_variance
identityИвStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_gammabatch_normalization_betabatch_normalization_moving_mean#batch_normalization_moving_variance*
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
GPU 2J 8В *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_17182
StatefulPartitionedCallН
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
ъ
Ж
N__inference_batch_normalization_1_layer_call_and_return_conditional_losses_890

inputs.
*readvariableop_batch_normalization_1_gamma/
+readvariableop_1_batch_normalization_1_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_1_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance
identityИвAssignNewValueвAssignNewValue_1З
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_1_gamma*
_output_shapes
: *
dtype02
ReadVariableOpМ
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_1_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1└
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp╩
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3▒
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_1_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_1/moving_mean*
_output_shapes
 *
dtype02
AssignNewValue╟
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_1/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1ж
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
зз
ё
F__inference_functional_1_layer_call_and_return_conditional_losses_2809

inputs
conv2d_conv2d_kernel1
-batch_normalization_batch_normalization_gamma0
,batch_normalization_batch_normalization_beta7
3batch_normalization_batch_normalization_moving_mean;
7batch_normalization_batch_normalization_moving_variance6
2depthwise_conv2d_depthwise_conv2d_depthwise_kernel5
1batch_normalization_1_batch_normalization_1_gamma4
0batch_normalization_1_batch_normalization_1_beta;
7batch_normalization_1_batch_normalization_1_moving_mean?
;batch_normalization_1_batch_normalization_1_moving_variance
conv2d_1_conv2d_1_kernel5
1batch_normalization_2_batch_normalization_2_gamma4
0batch_normalization_2_batch_normalization_2_beta;
7batch_normalization_2_batch_normalization_2_moving_mean?
;batch_normalization_2_batch_normalization_2_moving_variance:
6depthwise_conv2d_1_depthwise_conv2d_1_depthwise_kernel5
1batch_normalization_3_batch_normalization_3_gamma4
0batch_normalization_3_batch_normalization_3_beta;
7batch_normalization_3_batch_normalization_3_moving_mean?
;batch_normalization_3_batch_normalization_3_moving_variance
conv2d_2_conv2d_2_kernel5
1batch_normalization_4_batch_normalization_4_gamma4
0batch_normalization_4_batch_normalization_4_beta;
7batch_normalization_4_batch_normalization_4_moving_mean?
;batch_normalization_4_batch_normalization_4_moving_variance:
6depthwise_conv2d_2_depthwise_conv2d_2_depthwise_kernel5
1batch_normalization_5_batch_normalization_5_gamma4
0batch_normalization_5_batch_normalization_5_beta;
7batch_normalization_5_batch_normalization_5_moving_mean?
;batch_normalization_5_batch_normalization_5_moving_variance
conv2d_3_conv2d_3_kernel5
1batch_normalization_6_batch_normalization_6_gamma4
0batch_normalization_6_batch_normalization_6_beta;
7batch_normalization_6_batch_normalization_6_moving_mean?
;batch_normalization_6_batch_normalization_6_moving_variance:
6depthwise_conv2d_3_depthwise_conv2d_3_depthwise_kernel5
1batch_normalization_7_batch_normalization_7_gamma4
0batch_normalization_7_batch_normalization_7_beta;
7batch_normalization_7_batch_normalization_7_moving_mean?
;batch_normalization_7_batch_normalization_7_moving_variance
conv2d_4_conv2d_4_kernel5
1batch_normalization_8_batch_normalization_8_gamma4
0batch_normalization_8_batch_normalization_8_beta;
7batch_normalization_8_batch_normalization_8_moving_mean?
;batch_normalization_8_batch_normalization_8_moving_variance
dense_dense_kernel
dense_dense_bias
identityИв+batch_normalization/StatefulPartitionedCallв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallв-batch_normalization_3/StatefulPartitionedCallв-batch_normalization_4/StatefulPartitionedCallв-batch_normalization_5/StatefulPartitionedCallв-batch_normalization_6/StatefulPartitionedCallв-batch_normalization_7/StatefulPartitionedCallв-batch_normalization_8/StatefulPartitionedCallвconv2d/StatefulPartitionedCallв conv2d_1/StatefulPartitionedCallв conv2d_2/StatefulPartitionedCallв conv2d_3/StatefulPartitionedCallв conv2d_4/StatefulPartitionedCallвdense/StatefulPartitionedCallв(depthwise_conv2d/StatefulPartitionedCallв*depthwise_conv2d_1/StatefulPartitionedCallв*depthwise_conv2d_2/StatefulPartitionedCallв*depthwise_conv2d_3/StatefulPartitionedCall√
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
GPU 2J 8В *Y
fTRR
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_16752(
&tf_op_layer_ExpandDims/PartitionedCallж
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
GPU 2J 8В *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_16902 
conv2d/StatefulPartitionedCallА
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0-batch_normalization_batch_normalization_gamma,batch_normalization_batch_normalization_beta3batch_normalization_batch_normalization_moving_mean7batch_normalization_batch_normalization_moving_variance*
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
GPU 2J 8В *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_17362-
+batch_normalization/StatefulPartitionedCallЎ
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
GPU 2J 8В *H
fCRA
?__inference_re_lu_layer_call_and_return_conditional_losses_17652
re_lu/PartitionedCall╨
(depthwise_conv2d/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:02depthwise_conv2d_depthwise_conv2d_depthwise_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_depthwise_conv2d_layer_call_and_return_conditional_losses_8282*
(depthwise_conv2d/StatefulPartitionedCallа
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall1depthwise_conv2d/StatefulPartitionedCall:output:01batch_normalization_1_batch_normalization_1_gamma0batch_normalization_1_batch_normalization_1_beta7batch_normalization_1_batch_normalization_1_moving_mean;batch_normalization_1_batch_normalization_1_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_18162/
-batch_normalization_1/StatefulPartitionedCall■
re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_re_lu_1_layer_call_and_return_conditional_losses_18452
re_lu_1/PartitionedCallб
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall re_lu_1/PartitionedCall:output:0conv2d_1_conv2d_1_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_18602"
 conv2d_1/StatefulPartitionedCallШ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:01batch_normalization_2_batch_normalization_2_gamma0batch_normalization_2_batch_normalization_2_beta7batch_normalization_2_batch_normalization_2_moving_mean;batch_normalization_2_batch_normalization_2_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_19052/
-batch_normalization_2/StatefulPartitionedCall■
re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_re_lu_2_layer_call_and_return_conditional_losses_19342
re_lu_2/PartitionedCall▌
*depthwise_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall re_lu_2/PartitionedCall:output:06depthwise_conv2d_1_depthwise_conv2d_1_depthwise_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_depthwise_conv2d_1_layer_call_and_return_conditional_losses_10382,
*depthwise_conv2d_1/StatefulPartitionedCallв
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall3depthwise_conv2d_1/StatefulPartitionedCall:output:01batch_normalization_3_batch_normalization_3_gamma0batch_normalization_3_batch_normalization_3_beta7batch_normalization_3_batch_normalization_3_moving_mean;batch_normalization_3_batch_normalization_3_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_19852/
-batch_normalization_3/StatefulPartitionedCall■
re_lu_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_re_lu_3_layer_call_and_return_conditional_losses_20142
re_lu_3/PartitionedCallб
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall re_lu_3/PartitionedCall:output:0conv2d_2_conv2d_2_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_20292"
 conv2d_2/StatefulPartitionedCallШ
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:01batch_normalization_4_batch_normalization_4_gamma0batch_normalization_4_batch_normalization_4_beta7batch_normalization_4_batch_normalization_4_moving_mean;batch_normalization_4_batch_normalization_4_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_20742/
-batch_normalization_4/StatefulPartitionedCall■
re_lu_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_re_lu_4_layer_call_and_return_conditional_losses_21032
re_lu_4/PartitionedCall▌
*depthwise_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:06depthwise_conv2d_2_depthwise_conv2d_2_depthwise_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_depthwise_conv2d_2_layer_call_and_return_conditional_losses_12482,
*depthwise_conv2d_2/StatefulPartitionedCallв
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall3depthwise_conv2d_2/StatefulPartitionedCall:output:01batch_normalization_5_batch_normalization_5_gamma0batch_normalization_5_batch_normalization_5_beta7batch_normalization_5_batch_normalization_5_moving_mean;batch_normalization_5_batch_normalization_5_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_21542/
-batch_normalization_5/StatefulPartitionedCall■
re_lu_5/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_re_lu_5_layer_call_and_return_conditional_losses_21832
re_lu_5/PartitionedCallв
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_5/PartitionedCall:output:0conv2d_3_conv2d_3_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:А*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_21982"
 conv2d_3/StatefulPartitionedCallЩ
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:01batch_normalization_6_batch_normalization_6_gamma0batch_normalization_6_batch_normalization_6_beta7batch_normalization_6_batch_normalization_6_moving_mean;batch_normalization_6_batch_normalization_6_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:А*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_22432/
-batch_normalization_6/StatefulPartitionedCall 
re_lu_6/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_re_lu_6_layer_call_and_return_conditional_losses_22722
re_lu_6/PartitionedCall▐
*depthwise_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_6/PartitionedCall:output:06depthwise_conv2d_3_depthwise_conv2d_3_depthwise_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:А*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_depthwise_conv2d_3_layer_call_and_return_conditional_losses_14582,
*depthwise_conv2d_3/StatefulPartitionedCallг
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall3depthwise_conv2d_3/StatefulPartitionedCall:output:01batch_normalization_7_batch_normalization_7_gamma0batch_normalization_7_batch_normalization_7_beta7batch_normalization_7_batch_normalization_7_moving_mean;batch_normalization_7_batch_normalization_7_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:А*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_23232/
-batch_normalization_7/StatefulPartitionedCall 
re_lu_7/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_re_lu_7_layer_call_and_return_conditional_losses_23522
re_lu_7/PartitionedCallв
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall re_lu_7/PartitionedCall:output:0conv2d_4_conv2d_4_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:А*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_23672"
 conv2d_4/StatefulPartitionedCallЩ
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:01batch_normalization_8_batch_normalization_8_gamma0batch_normalization_8_batch_normalization_8_beta7batch_normalization_8_batch_normalization_8_moving_mean;batch_normalization_8_batch_normalization_8_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:А*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_24122/
-batch_normalization_8/StatefulPartitionedCall 
re_lu_8/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_re_lu_8_layer_call_and_return_conditional_losses_24412
re_lu_8/PartitionedCallФ
(global_average_pooling2d/PartitionedCallPartitionedCall re_lu_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_16622*
(global_average_pooling2d/PartitionedCallЄ
dropout/PartitionedCallPartitionedCall1global_average_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_24722
dropout/PartitionedCallЮ
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
GPU 2J 8В *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_24952
dense/StatefulPartitionedCallЮ
IdentityIdentity&dense/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall^dense/StatefulPartitionedCall)^depthwise_conv2d/StatefulPartitionedCall+^depthwise_conv2d_1/StatefulPartitionedCall+^depthwise_conv2d_2/StatefulPartitionedCall+^depthwise_conv2d_3/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*▀
_input_shapes═
╩:1(:::::::::::::::::::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2T
(depthwise_conv2d/StatefulPartitionedCall(depthwise_conv2d/StatefulPartitionedCall2X
*depthwise_conv2d_1/StatefulPartitionedCall*depthwise_conv2d_1/StatefulPartitionedCall2X
*depthwise_conv2d_2/StatefulPartitionedCall*depthwise_conv2d_2/StatefulPartitionedCall2X
*depthwise_conv2d_3/StatefulPartitionedCall*depthwise_conv2d_3/StatefulPartitionedCall:S O
+
_output_shapes
:         1(
 
_user_specified_nameinputs
Л
З
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_4782

inputs.
*readvariableop_batch_normalization_7_gamma/
+readvariableop_1_batch_normalization_7_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_7_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance
identityИвAssignNewValueвAssignNewValue_1И
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_7_gamma*
_output_shapes	
:А*
dtype02
ReadVariableOpН
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_7_beta*
_output_shapes	
:А*
dtype02
ReadVariableOp_1┴
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_7_moving_mean*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOp╦
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╘
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3:А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3▒
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_7_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_7/moving_mean*
_output_shapes
 *
dtype02
AssignNewValue╟
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_7/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1М
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*'
_output_shapes
:А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:А::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:O K
'
_output_shapes
:А
 
_user_specified_nameinputs
▀
А
4__inference_batch_normalization_4_layer_call_fn_4442

inputs
batch_normalization_4_gamma
batch_normalization_4_beta%
!batch_normalization_4_moving_mean)
%batch_normalization_4_moving_variance
identityИвStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_4_gammabatch_normalization_4_beta!batch_normalization_4_moving_mean%batch_normalization_4_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_20562
StatefulPartitionedCallН
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*5
_input_shapes$
":@::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:@
 
_user_specified_nameinputs
э
У
1__inference_depthwise_conv2d_1_layer_call_fn_1042

inputs'
#depthwise_conv2d_1_depthwise_kernel
identityИвStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinputs#depthwise_conv2d_1_depthwise_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_depthwise_conv2d_1_layer_call_and_return_conditional_losses_10382
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+                            :22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
 
З
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_4338

inputs.
*readvariableop_batch_normalization_3_gamma/
+readvariableop_1_batch_normalization_3_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_3_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance
identityИвAssignNewValueвAssignNewValue_1З
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_3_gamma*
_output_shapes
: *
dtype02
ReadVariableOpМ
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_3_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1└
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp╩
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╧
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3▒
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_3_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_3/moving_mean*
_output_shapes
 *
dtype02
AssignNewValue╟
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_3/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1Л
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:N J
&
_output_shapes
: 
 
_user_specified_nameinputs
ю
у
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2412

inputs.
*readvariableop_batch_normalization_8_gamma/
+readvariableop_1_batch_normalization_8_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_8_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance
identityИИ
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_8_gamma*
_output_shapes	
:А*
dtype02
ReadVariableOpН
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_8_beta*
_output_shapes	
:А*
dtype02
ReadVariableOp_1┴
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOp╦
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╞
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3:А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3h
IdentityIdentityFusedBatchNormV3:y:0*
T0*'
_output_shapes
:А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:А:::::O K
'
_output_shapes
:А
 
_user_specified_nameinputs
у
А
4__inference_batch_normalization_8_layer_call_fn_4940

inputs
batch_normalization_8_gamma
batch_normalization_8_beta%
!batch_normalization_8_moving_mean)
%batch_normalization_8_moving_variance
identityИвStatefulPartitionedCallЁ
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_8_gammabatch_normalization_8_beta!batch_normalization_8_moving_mean%batch_normalization_8_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_23942
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:А::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:А
 
_user_specified_nameinputs
█
у
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1639

inputs.
*readvariableop_batch_normalization_8_gamma/
+readvariableop_1_batch_normalization_8_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_8_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance
identityИИ
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_8_gamma*
_output_shapes	
:А*
dtype02
ReadVariableOpН
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_8_beta*
_output_shapes	
:А*
dtype02
ReadVariableOp_1┴
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOp╦
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1с
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Г
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,                           А:::::j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
ы
З
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_4469

inputs.
*readvariableop_batch_normalization_4_gamma/
+readvariableop_1_batch_normalization_4_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_4_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance
identityИвAssignNewValueвAssignNewValue_1З
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_4_gamma*
_output_shapes
:@*
dtype02
ReadVariableOpМ
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_4_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1└
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp╩
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3▒
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_4_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_4/moving_mean*
_output_shapes
 *
dtype02
AssignNewValue╟
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_4/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1ж
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
Фf
н
__inference__traced_save_5221
file_prefix,
(savev2_conv2d_kernel_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop@
<savev2_depthwise_conv2d_depthwise_kernel_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableopB
>savev2_depthwise_conv2d_1_depthwise_kernel_read_readvariableop:
6savev2_batch_normalization_3_gamma_read_readvariableop9
5savev2_batch_normalization_3_beta_read_readvariableop@
<savev2_batch_normalization_3_moving_mean_read_readvariableopD
@savev2_batch_normalization_3_moving_variance_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop:
6savev2_batch_normalization_4_gamma_read_readvariableop9
5savev2_batch_normalization_4_beta_read_readvariableop@
<savev2_batch_normalization_4_moving_mean_read_readvariableopD
@savev2_batch_normalization_4_moving_variance_read_readvariableopB
>savev2_depthwise_conv2d_2_depthwise_kernel_read_readvariableop:
6savev2_batch_normalization_5_gamma_read_readvariableop9
5savev2_batch_normalization_5_beta_read_readvariableop@
<savev2_batch_normalization_5_moving_mean_read_readvariableopD
@savev2_batch_normalization_5_moving_variance_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop:
6savev2_batch_normalization_6_gamma_read_readvariableop9
5savev2_batch_normalization_6_beta_read_readvariableop@
<savev2_batch_normalization_6_moving_mean_read_readvariableopD
@savev2_batch_normalization_6_moving_variance_read_readvariableopB
>savev2_depthwise_conv2d_3_depthwise_kernel_read_readvariableop:
6savev2_batch_normalization_7_gamma_read_readvariableop9
5savev2_batch_normalization_7_beta_read_readvariableop@
<savev2_batch_normalization_7_moving_mean_read_readvariableopD
@savev2_batch_normalization_7_moving_variance_read_readvariableop.
*savev2_conv2d_4_kernel_read_readvariableop:
6savev2_batch_normalization_8_gamma_read_readvariableop9
5savev2_batch_normalization_8_beta_read_readvariableop@
<savev2_batch_normalization_8_moving_mean_read_readvariableopD
@savev2_batch_normalization_8_moving_variance_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop
savev2_const

identity_1ИвMergeV2CheckpointsП
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
ConstН
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_2495c199091d4a4cade15864c9525800/part2	
Const_1Л
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
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameИ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*Ъ
valueРBН0B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-6/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-10/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-14/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-17/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-17/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-17/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesш
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*s
valuejBh0B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesЁ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop<savev2_depthwise_conv2d_depthwise_kernel_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop>savev2_depthwise_conv2d_1_depthwise_kernel_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop6savev2_batch_normalization_4_gamma_read_readvariableop5savev2_batch_normalization_4_beta_read_readvariableop<savev2_batch_normalization_4_moving_mean_read_readvariableop@savev2_batch_normalization_4_moving_variance_read_readvariableop>savev2_depthwise_conv2d_2_depthwise_kernel_read_readvariableop6savev2_batch_normalization_5_gamma_read_readvariableop5savev2_batch_normalization_5_beta_read_readvariableop<savev2_batch_normalization_5_moving_mean_read_readvariableop@savev2_batch_normalization_5_moving_variance_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop6savev2_batch_normalization_6_gamma_read_readvariableop5savev2_batch_normalization_6_beta_read_readvariableop<savev2_batch_normalization_6_moving_mean_read_readvariableop@savev2_batch_normalization_6_moving_variance_read_readvariableop>savev2_depthwise_conv2d_3_depthwise_kernel_read_readvariableop6savev2_batch_normalization_7_gamma_read_readvariableop5savev2_batch_normalization_7_beta_read_readvariableop<savev2_batch_normalization_7_moving_mean_read_readvariableop@savev2_batch_normalization_7_moving_variance_read_readvariableop*savev2_conv2d_4_kernel_read_readvariableop6savev2_batch_normalization_8_gamma_read_readvariableop5savev2_batch_normalization_8_beta_read_readvariableop<savev2_batch_normalization_8_moving_mean_read_readvariableop@savev2_batch_normalization_8_moving_variance_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *>
dtypes4
2202
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesб
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

identity_1Identity_1:output:0*┤
_input_shapesв
Я: :( : : : : : : : : : :  : : : : : : : : : : @:@:@:@:@:@:@:@:@:@:@А:А:А:А:А:А:А:А:А:А:АА:А:А:А:А:	А:: 2(
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
: : 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 	

_output_shapes
: : 


_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:-)
'
_output_shapes
:@А:! 

_output_shapes	
:А:!!

_output_shapes	
:А:!"

_output_shapes	
:А:!#

_output_shapes	
:А:-$)
'
_output_shapes
:А:!%

_output_shapes	
:А:!&

_output_shapes	
:А:!'

_output_shapes	
:А:!(

_output_shapes	
:А:.)*
(
_output_shapes
:АА:!*

_output_shapes	
:А:!+

_output_shapes	
:А:!,

_output_shapes	
:А:!-

_output_shapes	
:А:%.!

_output_shapes
:	А: /

_output_shapes
::0

_output_shapes
: 
 
З
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_4166

inputs.
*readvariableop_batch_normalization_2_gamma/
+readvariableop_1_batch_normalization_2_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_2_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance
identityИвAssignNewValueвAssignNewValue_1З
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_2_gamma*
_output_shapes
: *
dtype02
ReadVariableOpМ
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_2_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1└
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp╩
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╧
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3▒
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_2_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_2/moving_mean*
_output_shapes
 *
dtype02
AssignNewValue╟
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_2/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1Л
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:N J
&
_output_shapes
: 
 
_user_specified_nameinputs
╨
№
L__inference_batch_normalization_layer_call_and_return_conditional_losses_772

inputs,
(readvariableop_batch_normalization_gamma-
)readvariableop_1_batch_normalization_betaC
?fusedbatchnormv3_readvariableop_batch_normalization_moving_meanI
Efusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance
identityИвAssignNewValueвAssignNewValue_1Е
ReadVariableOpReadVariableOp(readvariableop_batch_normalization_gamma*
_output_shapes
: *
dtype02
ReadVariableOpК
ReadVariableOp_1ReadVariableOp)readvariableop_1_batch_normalization_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1╛
FusedBatchNormV3/ReadVariableOpReadVariableOp?fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp╚
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpEfusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3н
AssignNewValueAssignVariableOp?fusedbatchnormv3_readvariableop_batch_normalization_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization/moving_mean*
_output_shapes
 *
dtype02
AssignNewValue├
AssignNewValue_1AssignVariableOpEfusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*X
_classN
LJloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1ж
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
╧	
А
4__inference_batch_normalization_7_layer_call_fn_4863

inputs
batch_normalization_7_gamma
batch_normalization_7_beta%
!batch_normalization_7_moving_mean)
%batch_normalization_7_moving_variance
identityИвStatefulPartitionedCallЛ
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_7_gammabatch_normalization_7_beta!batch_normalization_7_moving_mean%batch_normalization_7_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_15202
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,                           А::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
╩	
А
4__inference_batch_normalization_1_layer_call_fn_4116

inputs
batch_normalization_1_gamma
batch_normalization_1_beta%
!batch_normalization_1_moving_mean)
%batch_normalization_1_moving_variance
identityИвStatefulPartitionedCallЙ
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_1_gammabatch_normalization_1_beta!batch_normalization_1_moving_mean%batch_normalization_1_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8902
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
°
_
&__inference_dropout_layer_call_fn_5035

inputs
identityИвStatefulPartitionedCall╧
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_24672
StatefulPartitionedCallЖ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	А2

Identity"
identityIdentity:output:0*
_input_shapes
:	А22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	А
 
_user_specified_nameinputs
Д
@
$__inference_re_lu_layer_call_fn_4017

inputs
identity╝
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
GPU 2J 8В *H
fCRA
?__inference_re_lu_layer_call_and_return_conditional_losses_17652
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
т
у
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2154

inputs.
*readvariableop_batch_normalization_5_gamma/
+readvariableop_1_batch_normalization_5_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_5_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance
identityИЗ
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_5_gamma*
_output_shapes
:@*
dtype02
ReadVariableOpМ
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_5_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1└
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_5_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp╩
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1┴
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3g
IdentityIdentityFusedBatchNormV3:y:0*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*5
_input_shapes$
":@:::::N J
&
_output_shapes
:@
 
_user_specified_nameinputs
╢	
Ў
2__inference_batch_normalization_layer_call_fn_3944

inputs
batch_normalization_gamma
batch_normalization_beta#
batch_normalization_moving_mean'
#batch_normalization_moving_variance
identityИвStatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_gammabatch_normalization_betabatch_normalization_moving_mean#batch_normalization_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_batch_normalization_layer_call_and_return_conditional_losses_7722
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
х
¤
M__inference_batch_normalization_layer_call_and_return_conditional_losses_1718

inputs,
(readvariableop_batch_normalization_gamma-
)readvariableop_1_batch_normalization_betaC
?fusedbatchnormv3_readvariableop_batch_normalization_moving_meanI
Efusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance
identityИвAssignNewValueвAssignNewValue_1Е
ReadVariableOpReadVariableOp(readvariableop_batch_normalization_gamma*
_output_shapes
: *
dtype02
ReadVariableOpК
ReadVariableOp_1ReadVariableOp)readvariableop_1_batch_normalization_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1╛
FusedBatchNormV3/ReadVariableOpReadVariableOp?fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp╚
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpEfusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╧
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3н
AssignNewValueAssignVariableOp?fusedbatchnormv3_readvariableop_batch_normalization_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization/moving_mean*
_output_shapes
 *
dtype02
AssignNewValue├
AssignNewValue_1AssignVariableOpEfusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*X
_classN
LJloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1Л
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
╜├
╢
F__inference_functional_1_layer_call_and_return_conditional_losses_3290
input_1.
*conv2d_conv2d_readvariableop_conv2d_kernel@
<batch_normalization_readvariableop_batch_normalization_gammaA
=batch_normalization_readvariableop_1_batch_normalization_betaW
Sbatch_normalization_fusedbatchnormv3_readvariableop_batch_normalization_moving_mean]
Ybatch_normalization_fusedbatchnormv3_readvariableop_1_batch_normalization_moving_varianceO
Kdepthwise_conv2d_depthwise_readvariableop_depthwise_conv2d_depthwise_kernelD
@batch_normalization_1_readvariableop_batch_normalization_1_gammaE
Abatch_normalization_1_readvariableop_1_batch_normalization_1_beta[
Wbatch_normalization_1_fusedbatchnormv3_readvariableop_batch_normalization_1_moving_meana
]batch_normalization_1_fusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance2
.conv2d_1_conv2d_readvariableop_conv2d_1_kernelD
@batch_normalization_2_readvariableop_batch_normalization_2_gammaE
Abatch_normalization_2_readvariableop_1_batch_normalization_2_beta[
Wbatch_normalization_2_fusedbatchnormv3_readvariableop_batch_normalization_2_moving_meana
]batch_normalization_2_fusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_varianceS
Odepthwise_conv2d_1_depthwise_readvariableop_depthwise_conv2d_1_depthwise_kernelD
@batch_normalization_3_readvariableop_batch_normalization_3_gammaE
Abatch_normalization_3_readvariableop_1_batch_normalization_3_beta[
Wbatch_normalization_3_fusedbatchnormv3_readvariableop_batch_normalization_3_moving_meana
]batch_normalization_3_fusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance2
.conv2d_2_conv2d_readvariableop_conv2d_2_kernelD
@batch_normalization_4_readvariableop_batch_normalization_4_gammaE
Abatch_normalization_4_readvariableop_1_batch_normalization_4_beta[
Wbatch_normalization_4_fusedbatchnormv3_readvariableop_batch_normalization_4_moving_meana
]batch_normalization_4_fusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_varianceS
Odepthwise_conv2d_2_depthwise_readvariableop_depthwise_conv2d_2_depthwise_kernelD
@batch_normalization_5_readvariableop_batch_normalization_5_gammaE
Abatch_normalization_5_readvariableop_1_batch_normalization_5_beta[
Wbatch_normalization_5_fusedbatchnormv3_readvariableop_batch_normalization_5_moving_meana
]batch_normalization_5_fusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance2
.conv2d_3_conv2d_readvariableop_conv2d_3_kernelD
@batch_normalization_6_readvariableop_batch_normalization_6_gammaE
Abatch_normalization_6_readvariableop_1_batch_normalization_6_beta[
Wbatch_normalization_6_fusedbatchnormv3_readvariableop_batch_normalization_6_moving_meana
]batch_normalization_6_fusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_varianceS
Odepthwise_conv2d_3_depthwise_readvariableop_depthwise_conv2d_3_depthwise_kernelD
@batch_normalization_7_readvariableop_batch_normalization_7_gammaE
Abatch_normalization_7_readvariableop_1_batch_normalization_7_beta[
Wbatch_normalization_7_fusedbatchnormv3_readvariableop_batch_normalization_7_moving_meana
]batch_normalization_7_fusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance2
.conv2d_4_conv2d_readvariableop_conv2d_4_kernelD
@batch_normalization_8_readvariableop_batch_normalization_8_gammaE
Abatch_normalization_8_readvariableop_1_batch_normalization_8_beta[
Wbatch_normalization_8_fusedbatchnormv3_readvariableop_batch_normalization_8_moving_meana
]batch_normalization_8_fusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance,
(dense_matmul_readvariableop_dense_kernel+
'dense_biasadd_readvariableop_dense_bias
identityИР
%tf_op_layer_ExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%tf_op_layer_ExpandDims/ExpandDims/dim═
!tf_op_layer_ExpandDims/ExpandDims
ExpandDimsinput_1.tf_op_layer_ExpandDims/ExpandDims/dim:output:0*
T0*
_cloned(*&
_output_shapes
:1(2#
!tf_op_layer_ExpandDims/ExpandDimsп
conv2d/Conv2D/ReadVariableOpReadVariableOp*conv2d_conv2d_readvariableop_conv2d_kernel*&
_output_shapes
:( *
dtype02
conv2d/Conv2D/ReadVariableOp╘
conv2d/Conv2DConv2D*tf_op_layer_ExpandDims/ExpandDims:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
2
conv2d/Conv2D┴
"batch_normalization/ReadVariableOpReadVariableOp<batch_normalization_readvariableop_batch_normalization_gamma*
_output_shapes
: *
dtype02$
"batch_normalization/ReadVariableOp╞
$batch_normalization/ReadVariableOp_1ReadVariableOp=batch_normalization_readvariableop_1_batch_normalization_beta*
_output_shapes
: *
dtype02&
$batch_normalization/ReadVariableOp_1·
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpSbatch_normalization_fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
_output_shapes
: *
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpД
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpYbatch_normalization_fusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
_output_shapes
: *
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1╔
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/Conv2D:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%oГ:*
is_training( 2&
$batch_normalization/FusedBatchNormV3~
re_lu/Relu6Relu6(batch_normalization/FusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2
re_lu/Relu6ъ
)depthwise_conv2d/depthwise/ReadVariableOpReadVariableOpKdepthwise_conv2d_depthwise_readvariableop_depthwise_conv2d_depthwise_kernel*&
_output_shapes
: *
dtype02+
)depthwise_conv2d/depthwise/ReadVariableOpЭ
 depthwise_conv2d/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 depthwise_conv2d/depthwise/Shapeе
(depthwise_conv2d/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2*
(depthwise_conv2d/depthwise/dilation_rate∙
depthwise_conv2d/depthwiseDepthwiseConv2dNativere_lu/Relu6:activations:01depthwise_conv2d/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
2
depthwise_conv2d/depthwise╔
$batch_normalization_1/ReadVariableOpReadVariableOp@batch_normalization_1_readvariableop_batch_normalization_1_gamma*
_output_shapes
: *
dtype02&
$batch_normalization_1/ReadVariableOp╬
&batch_normalization_1/ReadVariableOp_1ReadVariableOpAbatch_normalization_1_readvariableop_1_batch_normalization_1_beta*
_output_shapes
: *
dtype02(
&batch_normalization_1/ReadVariableOp_1В
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_1_fusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
: *
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpМ
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_1_fusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
_output_shapes
: *
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1т
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3#depthwise_conv2d/depthwise:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%oГ:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3Д
re_lu_1/Relu6Relu6*batch_normalization_1/FusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2
re_lu_1/Relu6╖
conv2d_1/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_conv2d_readvariableop_conv2d_1_kernel*&
_output_shapes
:  *
dtype02 
conv2d_1/Conv2D/ReadVariableOp╩
conv2d_1/Conv2DConv2Dre_lu_1/Relu6:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingSAME*
strides
2
conv2d_1/Conv2D╔
$batch_normalization_2/ReadVariableOpReadVariableOp@batch_normalization_2_readvariableop_batch_normalization_2_gamma*
_output_shapes
: *
dtype02&
$batch_normalization_2/ReadVariableOp╬
&batch_normalization_2/ReadVariableOp_1ReadVariableOpAbatch_normalization_2_readvariableop_1_batch_normalization_2_beta*
_output_shapes
: *
dtype02(
&batch_normalization_2/ReadVariableOp_1В
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_2_fusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
: *
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpМ
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_2_fusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
_output_shapes
: *
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1╫
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_1/Conv2D:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%oГ:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3Д
re_lu_2/Relu6Relu6*batch_normalization_2/FusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2
re_lu_2/Relu6Є
+depthwise_conv2d_1/depthwise/ReadVariableOpReadVariableOpOdepthwise_conv2d_1_depthwise_readvariableop_depthwise_conv2d_1_depthwise_kernel*&
_output_shapes
: *
dtype02-
+depthwise_conv2d_1/depthwise/ReadVariableOpб
"depthwise_conv2d_1/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"depthwise_conv2d_1/depthwise/Shapeй
*depthwise_conv2d_1/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2,
*depthwise_conv2d_1/depthwise/dilation_rateБ
depthwise_conv2d_1/depthwiseDepthwiseConv2dNativere_lu_2/Relu6:activations:03depthwise_conv2d_1/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
2
depthwise_conv2d_1/depthwise╔
$batch_normalization_3/ReadVariableOpReadVariableOp@batch_normalization_3_readvariableop_batch_normalization_3_gamma*
_output_shapes
: *
dtype02&
$batch_normalization_3/ReadVariableOp╬
&batch_normalization_3/ReadVariableOp_1ReadVariableOpAbatch_normalization_3_readvariableop_1_batch_normalization_3_beta*
_output_shapes
: *
dtype02(
&batch_normalization_3/ReadVariableOp_1В
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_3_fusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
_output_shapes
: *
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpМ
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_3_fusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
_output_shapes
: *
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ф
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3%depthwise_conv2d_1/depthwise:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%oГ:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3Д
re_lu_3/Relu6Relu6*batch_normalization_3/FusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2
re_lu_3/Relu6╖
conv2d_2/Conv2D/ReadVariableOpReadVariableOp.conv2d_2_conv2d_readvariableop_conv2d_2_kernel*&
_output_shapes
: @*
dtype02 
conv2d_2/Conv2D/ReadVariableOp╩
conv2d_2/Conv2DConv2Dre_lu_3/Relu6:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingSAME*
strides
2
conv2d_2/Conv2D╔
$batch_normalization_4/ReadVariableOpReadVariableOp@batch_normalization_4_readvariableop_batch_normalization_4_gamma*
_output_shapes
:@*
dtype02&
$batch_normalization_4/ReadVariableOp╬
&batch_normalization_4/ReadVariableOp_1ReadVariableOpAbatch_normalization_4_readvariableop_1_batch_normalization_4_beta*
_output_shapes
:@*
dtype02(
&batch_normalization_4/ReadVariableOp_1В
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_4_fusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
_output_shapes
:@*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpМ
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_4_fusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
_output_shapes
:@*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1╫
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_2/Conv2D:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:@:@:@:@:@:*
epsilon%oГ:*
is_training( 2(
&batch_normalization_4/FusedBatchNormV3Д
re_lu_4/Relu6Relu6*batch_normalization_4/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:@2
re_lu_4/Relu6Є
+depthwise_conv2d_2/depthwise/ReadVariableOpReadVariableOpOdepthwise_conv2d_2_depthwise_readvariableop_depthwise_conv2d_2_depthwise_kernel*&
_output_shapes
:@*
dtype02-
+depthwise_conv2d_2/depthwise/ReadVariableOpб
"depthwise_conv2d_2/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @      2$
"depthwise_conv2d_2/depthwise/Shapeй
*depthwise_conv2d_2/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2,
*depthwise_conv2d_2/depthwise/dilation_rateА
depthwise_conv2d_2/depthwiseDepthwiseConv2dNativere_lu_4/Relu6:activations:03depthwise_conv2d_2/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingSAME*
strides
2
depthwise_conv2d_2/depthwise╔
$batch_normalization_5/ReadVariableOpReadVariableOp@batch_normalization_5_readvariableop_batch_normalization_5_gamma*
_output_shapes
:@*
dtype02&
$batch_normalization_5/ReadVariableOp╬
&batch_normalization_5/ReadVariableOp_1ReadVariableOpAbatch_normalization_5_readvariableop_1_batch_normalization_5_beta*
_output_shapes
:@*
dtype02(
&batch_normalization_5/ReadVariableOp_1В
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_5_fusedbatchnormv3_readvariableop_batch_normalization_5_moving_mean*
_output_shapes
:@*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpМ
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_5_fusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance*
_output_shapes
:@*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ф
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3%depthwise_conv2d_2/depthwise:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:@:@:@:@:@:*
epsilon%oГ:*
is_training( 2(
&batch_normalization_5/FusedBatchNormV3Д
re_lu_5/Relu6Relu6*batch_normalization_5/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:@2
re_lu_5/Relu6╕
conv2d_3/Conv2D/ReadVariableOpReadVariableOp.conv2d_3_conv2d_readvariableop_conv2d_3_kernel*'
_output_shapes
:@А*
dtype02 
conv2d_3/Conv2D/ReadVariableOp╦
conv2d_3/Conv2DConv2Dre_lu_5/Relu6:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:А*
paddingSAME*
strides
2
conv2d_3/Conv2D╩
$batch_normalization_6/ReadVariableOpReadVariableOp@batch_normalization_6_readvariableop_batch_normalization_6_gamma*
_output_shapes	
:А*
dtype02&
$batch_normalization_6/ReadVariableOp╧
&batch_normalization_6/ReadVariableOp_1ReadVariableOpAbatch_normalization_6_readvariableop_1_batch_normalization_6_beta*
_output_shapes	
:А*
dtype02(
&batch_normalization_6/ReadVariableOp_1Г
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_6_fusedbatchnormv3_readvariableop_batch_normalization_6_moving_mean*
_output_shapes	
:А*
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpН
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_6_fusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance*
_output_shapes	
:А*
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1▄
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv2d_3/Conv2D:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3:А:А:А:А:А:*
epsilon%oГ:*
is_training( 2(
&batch_normalization_6/FusedBatchNormV3Е
re_lu_6/Relu6Relu6*batch_normalization_6/FusedBatchNormV3:y:0*
T0*'
_output_shapes
:А2
re_lu_6/Relu6є
+depthwise_conv2d_3/depthwise/ReadVariableOpReadVariableOpOdepthwise_conv2d_3_depthwise_readvariableop_depthwise_conv2d_3_depthwise_kernel*'
_output_shapes
:А*
dtype02-
+depthwise_conv2d_3/depthwise/ReadVariableOpб
"depthwise_conv2d_3/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      А      2$
"depthwise_conv2d_3/depthwise/Shapeй
*depthwise_conv2d_3/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2,
*depthwise_conv2d_3/depthwise/dilation_rateБ
depthwise_conv2d_3/depthwiseDepthwiseConv2dNativere_lu_6/Relu6:activations:03depthwise_conv2d_3/depthwise/ReadVariableOp:value:0*
T0*'
_output_shapes
:А*
paddingSAME*
strides
2
depthwise_conv2d_3/depthwise╩
$batch_normalization_7/ReadVariableOpReadVariableOp@batch_normalization_7_readvariableop_batch_normalization_7_gamma*
_output_shapes	
:А*
dtype02&
$batch_normalization_7/ReadVariableOp╧
&batch_normalization_7/ReadVariableOp_1ReadVariableOpAbatch_normalization_7_readvariableop_1_batch_normalization_7_beta*
_output_shapes	
:А*
dtype02(
&batch_normalization_7/ReadVariableOp_1Г
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_7_fusedbatchnormv3_readvariableop_batch_normalization_7_moving_mean*
_output_shapes	
:А*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpН
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_7_fusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance*
_output_shapes	
:А*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1щ
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3%depthwise_conv2d_3/depthwise:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3:А:А:А:А:А:*
epsilon%oГ:*
is_training( 2(
&batch_normalization_7/FusedBatchNormV3Е
re_lu_7/Relu6Relu6*batch_normalization_7/FusedBatchNormV3:y:0*
T0*'
_output_shapes
:А2
re_lu_7/Relu6╣
conv2d_4/Conv2D/ReadVariableOpReadVariableOp.conv2d_4_conv2d_readvariableop_conv2d_4_kernel*(
_output_shapes
:АА*
dtype02 
conv2d_4/Conv2D/ReadVariableOp╦
conv2d_4/Conv2DConv2Dre_lu_7/Relu6:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:А*
paddingSAME*
strides
2
conv2d_4/Conv2D╩
$batch_normalization_8/ReadVariableOpReadVariableOp@batch_normalization_8_readvariableop_batch_normalization_8_gamma*
_output_shapes	
:А*
dtype02&
$batch_normalization_8/ReadVariableOp╧
&batch_normalization_8/ReadVariableOp_1ReadVariableOpAbatch_normalization_8_readvariableop_1_batch_normalization_8_beta*
_output_shapes	
:А*
dtype02(
&batch_normalization_8/ReadVariableOp_1Г
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_8_fusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean*
_output_shapes	
:А*
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpН
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_8_fusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance*
_output_shapes	
:А*
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1▄
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3conv2d_4/Conv2D:output:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3:А:А:А:А:А:*
epsilon%oГ:*
is_training( 2(
&batch_normalization_8/FusedBatchNormV3Е
re_lu_8/Relu6Relu6*batch_normalization_8/FusedBatchNormV3:y:0*
T0*'
_output_shapes
:А2
re_lu_8/Relu6│
/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      21
/global_average_pooling2d/Mean/reduction_indices╟
global_average_pooling2d/MeanMeanre_lu_8/Relu6:activations:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*
_output_shapes
:	А2
global_average_pooling2d/MeanВ
dropout/IdentityIdentity&global_average_pooling2d/Mean:output:0*
T0*
_output_shapes
:	А2
dropout/Identityд
dense/MatMul/ReadVariableOpReadVariableOp(dense_matmul_readvariableop_dense_kernel*
_output_shapes
:	А*
dtype02
dense/MatMul/ReadVariableOpП
dense/MatMulMatMuldropout/Identity:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense/MatMulа
dense/BiasAdd/ReadVariableOpReadVariableOp'dense_biasadd_readvariableop_dense_bias*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOpР
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
identityIdentity:output:0*▀
_input_shapes═
╩:1(::::::::::::::::::::::::::::::::::::::::::::::::T P
+
_output_shapes
:         1(
!
_user_specified_name	input_1
 
З
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4035

inputs.
*readvariableop_batch_normalization_1_gamma/
+readvariableop_1_batch_normalization_1_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_1_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance
identityИвAssignNewValueвAssignNewValue_1З
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_1_gamma*
_output_shapes
: *
dtype02
ReadVariableOpМ
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_1_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1└
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp╩
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╧
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3▒
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_1_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_1/moving_mean*
_output_shapes
 *
dtype02
AssignNewValue╟
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_1/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1Л
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:N J
&
_output_shapes
: 
 
_user_specified_nameinputs
─
u
'__inference_conv2d_1_layer_call_fn_4148

inputs
conv2d_1_kernel
identityИвStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_18602
StatefulPartitionedCallН
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*)
_input_shapes
: :22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
: 
 
_user_specified_nameinputs
═	
А
4__inference_batch_normalization_3_layer_call_fn_4320

inputs
batch_normalization_3_gamma
batch_normalization_3_beta%
!batch_normalization_3_moving_mean)
%batch_normalization_3_moving_variance
identityИвStatefulPartitionedCallМ
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_3_gammabatch_normalization_3_beta!batch_normalization_3_moving_mean%batch_normalization_3_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_11272
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
▀
А
4__inference_batch_normalization_2_layer_call_fn_4193

inputs
batch_normalization_2_gamma
batch_normalization_2_beta%
!batch_normalization_2_moving_mean)
%batch_normalization_2_moving_variance
identityИвStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_2_gammabatch_normalization_2_beta!batch_normalization_2_moving_mean%batch_normalization_2_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_18872
StatefulPartitionedCallН
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": ::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
: 
 
_user_specified_nameinputs
 
З
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_4533

inputs.
*readvariableop_batch_normalization_5_gamma/
+readvariableop_1_batch_normalization_5_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_5_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance
identityИвAssignNewValueвAssignNewValue_1З
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_5_gamma*
_output_shapes
:@*
dtype02
ReadVariableOpМ
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_5_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1└
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_5_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp╩
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╧
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3▒
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_5_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_5/moving_mean*
_output_shapes
 *
dtype02
AssignNewValue╟
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_5/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1Л
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*5
_input_shapes$
":@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:N J
&
_output_shapes
:@
 
_user_specified_nameinputs
╦	
А
4__inference_batch_normalization_4_layer_call_fn_4496

inputs
batch_normalization_4_gamma
batch_normalization_4_beta%
!batch_normalization_4_moving_mean)
%batch_normalization_4_moving_variance
identityИвStatefulPartitionedCallК
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_4_gammabatch_normalization_4_beta!batch_normalization_4_moving_mean%batch_normalization_4_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_11922
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
ы
З
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_4284

inputs.
*readvariableop_batch_normalization_3_gamma/
+readvariableop_1_batch_normalization_3_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_3_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance
identityИвAssignNewValueвAssignNewValue_1З
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_3_gamma*
_output_shapes
: *
dtype02
ReadVariableOpМ
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_3_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1└
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp╩
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3▒
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_3_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_3/moving_mean*
_output_shapes
 *
dtype02
AssignNewValue╟
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_3/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1ж
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
т
у
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_4551

inputs.
*readvariableop_batch_normalization_5_gamma/
+readvariableop_1_batch_normalization_5_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_5_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance
identityИЗ
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_5_gamma*
_output_shapes
:@*
dtype02
ReadVariableOpМ
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_5_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1└
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_5_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp╩
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1┴
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3g
IdentityIdentityFusedBatchNormV3:y:0*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*5
_input_shapes$
":@:::::N J
&
_output_shapes
:@
 
_user_specified_nameinputs
╧
М
B__inference_conv2d_4_layer_call_and_return_conditional_losses_4889

inputs)
%conv2d_readvariableop_conv2d_4_kernel
identityИЮ
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_4_kernel*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOpЫ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:А*
paddingSAME*
strides
2
Conv2Dc
IdentityIdentityConv2D:output:0*
T0*'
_output_shapes
:А2

Identity"
identityIdentity:output:0**
_input_shapes
:А::O K
'
_output_shapes
:А
 
_user_specified_nameinputs
╤	
А
4__inference_batch_normalization_6_layer_call_fn_4700

inputs
batch_normalization_6_gamma
batch_normalization_6_beta%
!batch_normalization_6_moving_mean)
%batch_normalization_6_moving_variance
identityИвStatefulPartitionedCallН
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_6_gammabatch_normalization_6_beta!batch_normalization_6_moving_mean%batch_normalization_6_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_14292
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,                           А::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
ь
B
&__inference_dropout_layer_call_fn_5040

inputs
identity╖
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_24722
PartitionedCalld
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
:	А2

Identity"
identityIdentity:output:0*
_input_shapes
:	А:G C

_output_shapes
:	А
 
_user_specified_nameinputs
п
]
A__inference_re_lu_3_layer_call_and_return_conditional_losses_2014

inputs
identityP
Relu6Relu6inputs*
T0*&
_output_shapes
: 2
Relu6f
IdentityIdentityRelu6:activations:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*%
_input_shapes
: :N J
&
_output_shapes
: 
 
_user_specified_nameinputs
Ч	
н
L__inference_depthwise_conv2d_3_layer_call_and_return_conditional_losses_1458

inputs@
<depthwise_readvariableop_depthwise_conv2d_3_depthwise_kernel
identityИ║
depthwise/ReadVariableOpReadVariableOp<depthwise_readvariableop_depthwise_conv2d_3_depthwise_kernel*'
_output_shapes
:А*
dtype02
depthwise/ReadVariableOp{
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      А      2
depthwise/ShapeГ
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
depthwise/dilation_rate╬
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           А*
paddingSAME*
strides
2
	depthwiseБ
IdentityIdentitydepthwise:output:0*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:,                           А::j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
╧
у
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1127

inputs.
*readvariableop_batch_normalization_3_gamma/
+readvariableop_1_batch_normalization_3_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_3_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance
identityИЗ
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_3_gamma*
_output_shapes
: *
dtype02
ReadVariableOpМ
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_3_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1└
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp╩
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            :::::i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
│
]
A__inference_re_lu_7_layer_call_and_return_conditional_losses_2352

inputs
identityQ
Relu6Relu6inputs*
T0*'
_output_shapes
:А2
Relu6g
IdentityIdentityRelu6:activations:0*
T0*'
_output_shapes
:А2

Identity"
identityIdentity:output:0*&
_input_shapes
:А:O K
'
_output_shapes
:А
 
_user_specified_nameinputs
н
[
?__inference_re_lu_layer_call_and_return_conditional_losses_4012

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
И
B
&__inference_re_lu_4_layer_call_fn_4515

inputs
identity╛
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_re_lu_4_layer_call_and_return_conditional_losses_21032
PartitionedCallk
IdentityIdentityPartitionedCall:output:0*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*%
_input_shapes
:@:N J
&
_output_shapes
:@
 
_user_specified_nameinputs
И
B
&__inference_re_lu_3_layer_call_fn_4384

inputs
identity╛
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_re_lu_3_layer_call_and_return_conditional_losses_20142
PartitionedCallk
IdentityIdentityPartitionedCall:output:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*%
_input_shapes
: :N J
&
_output_shapes
: 
 
_user_specified_nameinputs
═	
А
4__inference_batch_normalization_4_layer_call_fn_4505

inputs
batch_normalization_4_gamma
batch_normalization_4_beta%
!batch_normalization_4_moving_mean)
%batch_normalization_4_moving_variance
identityИвStatefulPartitionedCallМ
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_4_gammabatch_normalization_4_beta!batch_normalization_4_moving_mean%batch_normalization_4_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_12192
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
╧	
А
4__inference_batch_normalization_8_layer_call_fn_4994

inputs
batch_normalization_8_gamma
batch_normalization_8_beta%
!batch_normalization_8_moving_mean)
%batch_normalization_8_moving_variance
identityИвStatefulPartitionedCallЛ
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_8_gammabatch_normalization_8_beta!batch_normalization_8_moving_mean%batch_normalization_8_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_16122
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,                           А::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
ы
З
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1100

inputs.
*readvariableop_batch_normalization_3_gamma/
+readvariableop_1_batch_normalization_3_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_3_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance
identityИвAssignNewValueвAssignNewValue_1З
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_3_gamma*
_output_shapes
: *
dtype02
ReadVariableOpМ
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_3_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1└
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp╩
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3▒
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_3_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_3/moving_mean*
_output_shapes
 *
dtype02
AssignNewValue╟
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_3/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1ж
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
╪

`
A__inference_dropout_layer_call_and_return_conditional_losses_2467

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Constk
dropout/MulMulinputsdropout/Const:output:0*
T0*
_output_shapes
:	А2
dropout/Mulo
dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   А   2
dropout/Shapeм
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
_output_shapes
:	А*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y╢
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	А2
dropout/GreaterEqualw
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	А2
dropout/Castr
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*
_output_shapes
:	А2
dropout/Mul_1]
IdentityIdentitydropout/Mul_1:z:0*
T0*
_output_shapes
:	А2

Identity"
identityIdentity:output:0*
_input_shapes
:	А:G C

_output_shapes
:	А
 
_user_specified_nameinputs
─
И
@__inference_conv2d_layer_call_and_return_conditional_losses_1690

inputs'
#conv2d_readvariableop_conv2d_kernel
identityИЪ
Conv2D/ReadVariableOpReadVariableOp#conv2d_readvariableop_conv2d_kernel*&
_output_shapes
:( *
dtype02
Conv2D/ReadVariableOpЫ
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
╧
у
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1337

inputs.
*readvariableop_batch_normalization_5_gamma/
+readvariableop_1_batch_normalization_5_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_5_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance
identityИЗ
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_5_gamma*
_output_shapes
:@*
dtype02
ReadVariableOpМ
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_5_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1└
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_5_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp╩
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           @:::::i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
ў
З
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1612

inputs.
*readvariableop_batch_normalization_8_gamma/
+readvariableop_1_batch_normalization_8_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_8_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance
identityИвAssignNewValueвAssignNewValue_1И
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_8_gamma*
_output_shapes	
:А*
dtype02
ReadVariableOpН
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_8_beta*
_output_shapes	
:А*
dtype02
ReadVariableOp_1┴
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOp╦
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1я
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3▒
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_8_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_8/moving_mean*
_output_shapes
 *
dtype02
AssignNewValue╟
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_8/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1з
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,                           А::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
п
]
A__inference_re_lu_5_layer_call_and_return_conditional_losses_4628

inputs
identityP
Relu6Relu6inputs*
T0*&
_output_shapes
:@2
Relu6f
IdentityIdentityRelu6:activations:0*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*%
_input_shapes
:@:N J
&
_output_shapes
:@
 
_user_specified_nameinputs
│
]
A__inference_re_lu_7_layer_call_and_return_conditional_losses_4877

inputs
identityQ
Relu6Relu6inputs*
T0*'
_output_shapes
:А2
Relu6g
IdentityIdentityRelu6:activations:0*
T0*'
_output_shapes
:А2

Identity"
identityIdentity:output:0*&
_input_shapes
:А:O K
'
_output_shapes
:А
 
_user_specified_nameinputs
т
у
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_4356

inputs.
*readvariableop_batch_normalization_3_gamma/
+readvariableop_1_batch_normalization_3_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_3_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance
identityИЗ
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_3_gamma*
_output_shapes
: *
dtype02
ReadVariableOpМ
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_3_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1└
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp╩
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1┴
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3g
IdentityIdentityFusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": :::::N J
&
_output_shapes
: 
 
_user_specified_nameinputs
═
Ў
2__inference_batch_normalization_layer_call_fn_4007

inputs
batch_normalization_gamma
batch_normalization_beta#
batch_normalization_moving_mean'
#batch_normalization_moving_variance
identityИвStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_gammabatch_normalization_betabatch_normalization_moving_mean#batch_normalization_moving_variance*
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
GPU 2J 8В *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_17362
StatefulPartitionedCallН
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
ю
у
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2323

inputs.
*readvariableop_batch_normalization_7_gamma/
+readvariableop_1_batch_normalization_7_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_7_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance
identityИИ
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_7_gamma*
_output_shapes	
:А*
dtype02
ReadVariableOpН
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_7_beta*
_output_shapes	
:А*
dtype02
ReadVariableOp_1┴
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_7_moving_mean*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOp╦
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╞
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3:А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3h
IdentityIdentityFusedBatchNormV3:y:0*
T0*'
_output_shapes
:А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:А:::::O K
'
_output_shapes
:А
 
_user_specified_nameinputs
Ы 
│
+__inference_functional_1_layer_call_fn_3394
input_1
conv2d_kernel
batch_normalization_gamma
batch_normalization_beta#
batch_normalization_moving_mean'
#batch_normalization_moving_variance%
!depthwise_conv2d_depthwise_kernel
batch_normalization_1_gamma
batch_normalization_1_beta%
!batch_normalization_1_moving_mean)
%batch_normalization_1_moving_variance
conv2d_1_kernel
batch_normalization_2_gamma
batch_normalization_2_beta%
!batch_normalization_2_moving_mean)
%batch_normalization_2_moving_variance'
#depthwise_conv2d_1_depthwise_kernel
batch_normalization_3_gamma
batch_normalization_3_beta%
!batch_normalization_3_moving_mean)
%batch_normalization_3_moving_variance
conv2d_2_kernel
batch_normalization_4_gamma
batch_normalization_4_beta%
!batch_normalization_4_moving_mean)
%batch_normalization_4_moving_variance'
#depthwise_conv2d_2_depthwise_kernel
batch_normalization_5_gamma
batch_normalization_5_beta%
!batch_normalization_5_moving_mean)
%batch_normalization_5_moving_variance
conv2d_3_kernel
batch_normalization_6_gamma
batch_normalization_6_beta%
!batch_normalization_6_moving_mean)
%batch_normalization_6_moving_variance'
#depthwise_conv2d_3_depthwise_kernel
batch_normalization_7_gamma
batch_normalization_7_beta%
!batch_normalization_7_moving_mean)
%batch_normalization_7_moving_variance
conv2d_4_kernel
batch_normalization_8_gamma
batch_normalization_8_beta%
!batch_normalization_8_moving_mean)
%batch_normalization_8_moving_variance
dense_kernel

dense_bias
identityИвStatefulPartitionedCall╞
StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_kernelbatch_normalization_gammabatch_normalization_betabatch_normalization_moving_mean#batch_normalization_moving_variance!depthwise_conv2d_depthwise_kernelbatch_normalization_1_gammabatch_normalization_1_beta!batch_normalization_1_moving_mean%batch_normalization_1_moving_varianceconv2d_1_kernelbatch_normalization_2_gammabatch_normalization_2_beta!batch_normalization_2_moving_mean%batch_normalization_2_moving_variance#depthwise_conv2d_1_depthwise_kernelbatch_normalization_3_gammabatch_normalization_3_beta!batch_normalization_3_moving_mean%batch_normalization_3_moving_varianceconv2d_2_kernelbatch_normalization_4_gammabatch_normalization_4_beta!batch_normalization_4_moving_mean%batch_normalization_4_moving_variance#depthwise_conv2d_2_depthwise_kernelbatch_normalization_5_gammabatch_normalization_5_beta!batch_normalization_5_moving_mean%batch_normalization_5_moving_varianceconv2d_3_kernelbatch_normalization_6_gammabatch_normalization_6_beta!batch_normalization_6_moving_mean%batch_normalization_6_moving_variance#depthwise_conv2d_3_depthwise_kernelbatch_normalization_7_gammabatch_normalization_7_beta!batch_normalization_7_moving_mean%batch_normalization_7_moving_varianceconv2d_4_kernelbatch_normalization_8_gammabatch_normalization_8_beta!batch_normalization_8_moving_mean%batch_normalization_8_moving_variancedense_kernel
dense_bias*;
Tin4
220*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*Q
_read_only_resource_inputs3
1/	
 !"#$%&'()*+,-./*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_functional_1_layer_call_and_return_conditional_losses_28092
StatefulPartitionedCallЕ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*ш
_input_shapes╓
╙:         1(:::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:         1(
!
_user_specified_name	input_1
│
]
A__inference_re_lu_8_layer_call_and_return_conditional_losses_2441

inputs
identityQ
Relu6Relu6inputs*
T0*'
_output_shapes
:А2
Relu6g
IdentityIdentityRelu6:activations:0*
T0*'
_output_shapes
:А2

Identity"
identityIdentity:output:0*&
_input_shapes
:А:O K
'
_output_shapes
:А
 
_user_specified_nameinputs
у
А
4__inference_batch_normalization_7_layer_call_fn_4809

inputs
batch_normalization_7_gamma
batch_normalization_7_beta%
!batch_normalization_7_moving_mean)
%batch_normalization_7_moving_variance
identityИвStatefulPartitionedCallЁ
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_7_gammabatch_normalization_7_beta!batch_normalization_7_moving_mean%batch_normalization_7_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_23052
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:А::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:А
 
_user_specified_nameinputs
╬
т
N__inference_batch_normalization_1_layer_call_and_return_conditional_losses_917

inputs.
*readvariableop_batch_normalization_1_gamma/
+readvariableop_1_batch_normalization_1_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_1_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance
identityИЗ
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_1_gamma*
_output_shapes
: *
dtype02
ReadVariableOpМ
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_1_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1└
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp╩
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            :::::i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
й
н
?__inference_dense_layer_call_and_return_conditional_losses_2495

inputs&
"matmul_readvariableop_dense_kernel%
!biasadd_readvariableop_dense_bias
identityИТ
MatMul/ReadVariableOpReadVariableOp"matmul_readvariableop_dense_kernel*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOpj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
MatMulО
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
identityIdentity:output:0*&
_input_shapes
:	А:::G C

_output_shapes
:	А
 
_user_specified_nameinputs
╧
у
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_4487

inputs.
*readvariableop_batch_normalization_4_gamma/
+readvariableop_1_batch_normalization_4_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_4_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance
identityИЗ
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_4_gamma*
_output_shapes
:@*
dtype02
ReadVariableOpМ
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_4_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1└
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp╩
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           @:::::i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
т
у
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4053

inputs.
*readvariableop_batch_normalization_1_gamma/
+readvariableop_1_batch_normalization_1_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_1_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance
identityИЗ
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_1_gamma*
_output_shapes
: *
dtype02
ReadVariableOpМ
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_1_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1└
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp╩
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1┴
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3g
IdentityIdentityFusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": :::::N J
&
_output_shapes
: 
 
_user_specified_nameinputs
╧
у
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_4605

inputs.
*readvariableop_batch_normalization_5_gamma/
+readvariableop_1_batch_normalization_5_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_5_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance
identityИЗ
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_5_gamma*
_output_shapes
:@*
dtype02
ReadVariableOpМ
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_5_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1└
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_5_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp╩
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           @:::::i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
э
У
1__inference_depthwise_conv2d_2_layer_call_fn_1252

inputs'
#depthwise_conv2d_2_depthwise_kernel
identityИвStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinputs#depthwise_conv2d_2_depthwise_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_depthwise_conv2d_2_layer_call_and_return_conditional_losses_12482
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+                           @:22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
▀
А
4__inference_batch_normalization_3_layer_call_fn_4365

inputs
batch_normalization_3_gamma
batch_normalization_3_beta%
!batch_normalization_3_moving_mean)
%batch_normalization_3_moving_variance
identityИвStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_3_gammabatch_normalization_3_beta!batch_normalization_3_moving_mean%batch_normalization_3_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_19672
StatefulPartitionedCallН
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": ::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
: 
 
_user_specified_nameinputs
▀
А
4__inference_batch_normalization_5_layer_call_fn_4560

inputs
batch_normalization_5_gamma
batch_normalization_5_beta%
!batch_normalization_5_moving_mean)
%batch_normalization_5_moving_variance
identityИвStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_5_gammabatch_normalization_5_beta!batch_normalization_5_moving_mean%batch_normalization_5_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_21362
StatefulPartitionedCallН
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*5
_input_shapes$
":@::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:@
 
_user_specified_nameinputs
╪
к
"__inference_signature_wrapper_2913
input_1
conv2d_kernel
batch_normalization_gamma
batch_normalization_beta#
batch_normalization_moving_mean'
#batch_normalization_moving_variance%
!depthwise_conv2d_depthwise_kernel
batch_normalization_1_gamma
batch_normalization_1_beta%
!batch_normalization_1_moving_mean)
%batch_normalization_1_moving_variance
conv2d_1_kernel
batch_normalization_2_gamma
batch_normalization_2_beta%
!batch_normalization_2_moving_mean)
%batch_normalization_2_moving_variance'
#depthwise_conv2d_1_depthwise_kernel
batch_normalization_3_gamma
batch_normalization_3_beta%
!batch_normalization_3_moving_mean)
%batch_normalization_3_moving_variance
conv2d_2_kernel
batch_normalization_4_gamma
batch_normalization_4_beta%
!batch_normalization_4_moving_mean)
%batch_normalization_4_moving_variance'
#depthwise_conv2d_2_depthwise_kernel
batch_normalization_5_gamma
batch_normalization_5_beta%
!batch_normalization_5_moving_mean)
%batch_normalization_5_moving_variance
conv2d_3_kernel
batch_normalization_6_gamma
batch_normalization_6_beta%
!batch_normalization_6_moving_mean)
%batch_normalization_6_moving_variance'
#depthwise_conv2d_3_depthwise_kernel
batch_normalization_7_gamma
batch_normalization_7_beta%
!batch_normalization_7_moving_mean)
%batch_normalization_7_moving_variance
conv2d_4_kernel
batch_normalization_8_gamma
batch_normalization_8_beta%
!batch_normalization_8_moving_mean)
%batch_normalization_8_moving_variance
dense_kernel

dense_bias
identityИвStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_kernelbatch_normalization_gammabatch_normalization_betabatch_normalization_moving_mean#batch_normalization_moving_variance!depthwise_conv2d_depthwise_kernelbatch_normalization_1_gammabatch_normalization_1_beta!batch_normalization_1_moving_mean%batch_normalization_1_moving_varianceconv2d_1_kernelbatch_normalization_2_gammabatch_normalization_2_beta!batch_normalization_2_moving_mean%batch_normalization_2_moving_variance#depthwise_conv2d_1_depthwise_kernelbatch_normalization_3_gammabatch_normalization_3_beta!batch_normalization_3_moving_mean%batch_normalization_3_moving_varianceconv2d_2_kernelbatch_normalization_4_gammabatch_normalization_4_beta!batch_normalization_4_moving_mean%batch_normalization_4_moving_variance#depthwise_conv2d_2_depthwise_kernelbatch_normalization_5_gammabatch_normalization_5_beta!batch_normalization_5_moving_mean%batch_normalization_5_moving_varianceconv2d_3_kernelbatch_normalization_6_gammabatch_normalization_6_beta!batch_normalization_6_moving_mean%batch_normalization_6_moving_variance#depthwise_conv2d_3_depthwise_kernelbatch_normalization_7_gammabatch_normalization_7_beta!batch_normalization_7_moving_mean%batch_normalization_7_moving_varianceconv2d_4_kernelbatch_normalization_8_gammabatch_normalization_8_beta!batch_normalization_8_moving_mean%batch_normalization_8_moving_variancedense_kernel
dense_bias*;
Tin4
220*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*Q
_read_only_resource_inputs3
1/	
 !"#$%&'()*+,-./*-
config_proto

CPU

GPU 2J 8В *'
f"R 
__inference__wrapped_model_7142
StatefulPartitionedCallЕ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*▀
_input_shapes═
╩:1(:::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:K G
"
_output_shapes
:1(
!
_user_specified_name	input_1
п
]
A__inference_re_lu_1_layer_call_and_return_conditional_losses_4130

inputs
identityP
Relu6Relu6inputs*
T0*&
_output_shapes
: 2
Relu6f
IdentityIdentityRelu6:activations:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*%
_input_shapes
: :N J
&
_output_shapes
: 
 
_user_specified_nameinputs
╤	
А
4__inference_batch_normalization_8_layer_call_fn_5003

inputs
batch_normalization_8_gamma
batch_normalization_8_beta%
!batch_normalization_8_moving_mean)
%batch_normalization_8_moving_variance
identityИвStatefulPartitionedCallН
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_8_gammabatch_normalization_8_beta!batch_normalization_8_moving_mean%batch_normalization_8_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_16392
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,                           А::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
п
]
A__inference_re_lu_1_layer_call_and_return_conditional_losses_1845

inputs
identityP
Relu6Relu6inputs*
T0*&
_output_shapes
: 2
Relu6f
IdentityIdentityRelu6:activations:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*%
_input_shapes
: :N J
&
_output_shapes
: 
 
_user_specified_nameinputs
│
]
A__inference_re_lu_8_layer_call_and_return_conditional_losses_5008

inputs
identityQ
Relu6Relu6inputs*
T0*'
_output_shapes
:А2
Relu6g
IdentityIdentityRelu6:activations:0*
T0*'
_output_shapes
:А2

Identity"
identityIdentity:output:0*&
_input_shapes
:А:O K
'
_output_shapes
:А
 
_user_specified_nameinputs
ы
З
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_4220

inputs.
*readvariableop_batch_normalization_2_gamma/
+readvariableop_1_batch_normalization_2_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_2_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance
identityИвAssignNewValueвAssignNewValue_1З
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_2_gamma*
_output_shapes
: *
dtype02
ReadVariableOpМ
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_2_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1└
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp╩
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3▒
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_2_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_2/moving_mean*
_output_shapes
 *
dtype02
AssignNewValue╟
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_2/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1ж
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
╪

`
A__inference_dropout_layer_call_and_return_conditional_losses_5025

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Constk
dropout/MulMulinputsdropout/Const:output:0*
T0*
_output_shapes
:	А2
dropout/Mulo
dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   А   2
dropout/Shapeм
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
_output_shapes
:	А*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y╢
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	А2
dropout/GreaterEqualw
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	А2
dropout/Castr
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*
_output_shapes
:	А2
dropout/Mul_1]
IdentityIdentitydropout/Mul_1:z:0*
T0*
_output_shapes
:	А2

Identity"
identityIdentity:output:0*
_input_shapes
:	А:G C

_output_shapes
:	А
 
_user_specified_nameinputs
╕
n
R__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_1653

inputs
identityБ
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:                  2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:                  2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
У	
н
L__inference_depthwise_conv2d_1_layer_call_and_return_conditional_losses_1038

inputs@
<depthwise_readvariableop_depthwise_conv2d_1_depthwise_kernel
identityИ╣
depthwise/ReadVariableOpReadVariableOp<depthwise_readvariableop_depthwise_conv2d_1_depthwise_kernel*&
_output_shapes
: *
dtype02
depthwise/ReadVariableOp{
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2
depthwise/ShapeГ
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
depthwise/dilation_rate╬
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingVALID*
strides
2
	depthwiseА
IdentityIdentitydepthwise:output:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+                            ::i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
╥
l
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_1675

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dimЗ

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
█
у
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_4854

inputs.
*readvariableop_batch_normalization_7_gamma/
+readvariableop_1_batch_normalization_7_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_7_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance
identityИИ
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_7_gamma*
_output_shapes	
:А*
dtype02
ReadVariableOpН
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_7_beta*
_output_shapes	
:А*
dtype02
ReadVariableOp_1┴
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_7_moving_mean*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOp╦
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1с
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Г
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,                           А:::::j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
╞
u
'__inference_conv2d_3_layer_call_fn_4646

inputs
conv2d_3_kernel
identityИвStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_3_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:А*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_21982
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:А2

Identity"
identityIdentity:output:0*)
_input_shapes
:@:22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:@
 
_user_specified_nameinputs
Ш 
▓
+__inference_functional_1_layer_call_fn_3875

inputs
conv2d_kernel
batch_normalization_gamma
batch_normalization_beta#
batch_normalization_moving_mean'
#batch_normalization_moving_variance%
!depthwise_conv2d_depthwise_kernel
batch_normalization_1_gamma
batch_normalization_1_beta%
!batch_normalization_1_moving_mean)
%batch_normalization_1_moving_variance
conv2d_1_kernel
batch_normalization_2_gamma
batch_normalization_2_beta%
!batch_normalization_2_moving_mean)
%batch_normalization_2_moving_variance'
#depthwise_conv2d_1_depthwise_kernel
batch_normalization_3_gamma
batch_normalization_3_beta%
!batch_normalization_3_moving_mean)
%batch_normalization_3_moving_variance
conv2d_2_kernel
batch_normalization_4_gamma
batch_normalization_4_beta%
!batch_normalization_4_moving_mean)
%batch_normalization_4_moving_variance'
#depthwise_conv2d_2_depthwise_kernel
batch_normalization_5_gamma
batch_normalization_5_beta%
!batch_normalization_5_moving_mean)
%batch_normalization_5_moving_variance
conv2d_3_kernel
batch_normalization_6_gamma
batch_normalization_6_beta%
!batch_normalization_6_moving_mean)
%batch_normalization_6_moving_variance'
#depthwise_conv2d_3_depthwise_kernel
batch_normalization_7_gamma
batch_normalization_7_beta%
!batch_normalization_7_moving_mean)
%batch_normalization_7_moving_variance
conv2d_4_kernel
batch_normalization_8_gamma
batch_normalization_8_beta%
!batch_normalization_8_moving_mean)
%batch_normalization_8_moving_variance
dense_kernel

dense_bias
identityИвStatefulPartitionedCall┼
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_kernelbatch_normalization_gammabatch_normalization_betabatch_normalization_moving_mean#batch_normalization_moving_variance!depthwise_conv2d_depthwise_kernelbatch_normalization_1_gammabatch_normalization_1_beta!batch_normalization_1_moving_mean%batch_normalization_1_moving_varianceconv2d_1_kernelbatch_normalization_2_gammabatch_normalization_2_beta!batch_normalization_2_moving_mean%batch_normalization_2_moving_variance#depthwise_conv2d_1_depthwise_kernelbatch_normalization_3_gammabatch_normalization_3_beta!batch_normalization_3_moving_mean%batch_normalization_3_moving_varianceconv2d_2_kernelbatch_normalization_4_gammabatch_normalization_4_beta!batch_normalization_4_moving_mean%batch_normalization_4_moving_variance#depthwise_conv2d_2_depthwise_kernelbatch_normalization_5_gammabatch_normalization_5_beta!batch_normalization_5_moving_mean%batch_normalization_5_moving_varianceconv2d_3_kernelbatch_normalization_6_gammabatch_normalization_6_beta!batch_normalization_6_moving_mean%batch_normalization_6_moving_variance#depthwise_conv2d_3_depthwise_kernelbatch_normalization_7_gammabatch_normalization_7_beta!batch_normalization_7_moving_mean%batch_normalization_7_moving_varianceconv2d_4_kernelbatch_normalization_8_gammabatch_normalization_8_beta!batch_normalization_8_moving_mean%batch_normalization_8_moving_variancedense_kernel
dense_bias*;
Tin4
220*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*Q
_read_only_resource_inputs3
1/	
 !"#$%&'()*+,-./*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_functional_1_layer_call_and_return_conditional_losses_28092
StatefulPartitionedCallЕ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*ш
_input_shapes╓
╙:         1(:::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         1(
 
_user_specified_nameinputs
У	
н
L__inference_depthwise_conv2d_1_layer_call_and_return_conditional_losses_1026

inputs@
<depthwise_readvariableop_depthwise_conv2d_1_depthwise_kernel
identityИ╣
depthwise/ReadVariableOpReadVariableOp<depthwise_readvariableop_depthwise_conv2d_1_depthwise_kernel*&
_output_shapes
: *
dtype02
depthwise/ReadVariableOp{
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2
depthwise/ShapeГ
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
depthwise/dilation_rate╬
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingVALID*
strides
2
	depthwiseА
IdentityIdentitydepthwise:output:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+                            ::i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
И
B
&__inference_re_lu_1_layer_call_fn_4135

inputs
identity╛
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_re_lu_1_layer_call_and_return_conditional_losses_18452
PartitionedCallk
IdentityIdentityPartitionedCall:output:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*%
_input_shapes
: :N J
&
_output_shapes
: 
 
_user_specified_nameinputs
╦	
А
4__inference_batch_normalization_5_layer_call_fn_4614

inputs
batch_normalization_5_gamma
batch_normalization_5_beta%
!batch_normalization_5_moving_mean)
%batch_normalization_5_moving_variance
identityИвStatefulPartitionedCallК
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_5_gammabatch_normalization_5_beta!batch_normalization_5_moving_mean%batch_normalization_5_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_13102
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
ю
у
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_4931

inputs.
*readvariableop_batch_normalization_8_gamma/
+readvariableop_1_batch_normalization_8_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_8_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance
identityИИ
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_8_gamma*
_output_shapes	
:А*
dtype02
ReadVariableOpН
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_8_beta*
_output_shapes	
:А*
dtype02
ReadVariableOp_1┴
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOp╦
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╞
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3:А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3h
IdentityIdentityFusedBatchNormV3:y:0*
T0*'
_output_shapes
:А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:А:::::O K
'
_output_shapes
:А
 
_user_specified_nameinputs
с
А
4__inference_batch_normalization_4_layer_call_fn_4451

inputs
batch_normalization_4_gamma
batch_normalization_4_beta%
!batch_normalization_4_moving_mean)
%batch_normalization_4_moving_variance
identityИвStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_4_gammabatch_normalization_4_beta!batch_normalization_4_moving_mean%batch_normalization_4_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_20742
StatefulPartitionedCallН
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*5
_input_shapes$
":@::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:@
 
_user_specified_nameinputs
Л
З
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2305

inputs.
*readvariableop_batch_normalization_7_gamma/
+readvariableop_1_batch_normalization_7_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_7_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance
identityИвAssignNewValueвAssignNewValue_1И
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_7_gamma*
_output_shapes	
:А*
dtype02
ReadVariableOpН
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_7_beta*
_output_shapes	
:А*
dtype02
ReadVariableOp_1┴
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_7_moving_mean*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOp╦
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╘
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3:А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3▒
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_7_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_7/moving_mean*
_output_shapes
 *
dtype02
AssignNewValue╟
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_7/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1М
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*'
_output_shapes
:А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:А::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:O K
'
_output_shapes
:А
 
_user_specified_nameinputs
█
у
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1429

inputs.
*readvariableop_batch_normalization_6_gamma/
+readvariableop_1_batch_normalization_6_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_6_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance
identityИИ
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_6_gamma*
_output_shapes	
:А*
dtype02
ReadVariableOpН
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_6_beta*
_output_shapes	
:А*
dtype02
ReadVariableOp_1┴
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_6_moving_mean*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOp╦
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1с
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Г
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,                           А:::::j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
╧	
А
4__inference_batch_normalization_6_layer_call_fn_4691

inputs
batch_normalization_6_gamma
batch_normalization_6_beta%
!batch_normalization_6_moving_mean)
%batch_normalization_6_moving_variance
identityИвStatefulPartitionedCallЛ
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_6_gammabatch_normalization_6_beta!batch_normalization_6_moving_mean%batch_normalization_6_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_14022
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,                           А::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
ў
З
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1520

inputs.
*readvariableop_batch_normalization_7_gamma/
+readvariableop_1_batch_normalization_7_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_7_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance
identityИвAssignNewValueвAssignNewValue_1И
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_7_gamma*
_output_shapes	
:А*
dtype02
ReadVariableOpН
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_7_beta*
_output_shapes	
:А*
dtype02
ReadVariableOp_1┴
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_7_moving_mean*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOp╦
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1я
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3▒
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_7_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_7/moving_mean*
_output_shapes
 *
dtype02
AssignNewValue╟
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_7/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1з
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,                           А::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
т
у
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1985

inputs.
*readvariableop_batch_normalization_3_gamma/
+readvariableop_1_batch_normalization_3_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_3_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance
identityИЗ
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_3_gamma*
_output_shapes
: *
dtype02
ReadVariableOpМ
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_3_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1└
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp╩
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1┴
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3g
IdentityIdentityFusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": :::::N J
&
_output_shapes
: 
 
_user_specified_nameinputs
с
А
4__inference_batch_normalization_3_layer_call_fn_4374

inputs
batch_normalization_3_gamma
batch_normalization_3_beta%
!batch_normalization_3_moving_mean)
%batch_normalization_3_moving_variance
identityИвStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_3_gammabatch_normalization_3_beta!batch_normalization_3_moving_mean%batch_normalization_3_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_19852
StatefulPartitionedCallН
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": ::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
: 
 
_user_specified_nameinputs
Л
З
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_4913

inputs.
*readvariableop_batch_normalization_8_gamma/
+readvariableop_1_batch_normalization_8_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_8_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance
identityИвAssignNewValueвAssignNewValue_1И
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_8_gamma*
_output_shapes	
:А*
dtype02
ReadVariableOpН
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_8_beta*
_output_shapes	
:А*
dtype02
ReadVariableOp_1┴
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOp╦
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╘
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3:А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3▒
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_8_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_8/moving_mean*
_output_shapes
 *
dtype02
AssignNewValue╟
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_8/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1М
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*'
_output_shapes
:А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:А::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:O K
'
_output_shapes
:А
 
_user_specified_nameinputs
Л
З
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_4718

inputs.
*readvariableop_batch_normalization_6_gamma/
+readvariableop_1_batch_normalization_6_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_6_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance
identityИвAssignNewValueвAssignNewValue_1И
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_6_gamma*
_output_shapes	
:А*
dtype02
ReadVariableOpН
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_6_beta*
_output_shapes	
:А*
dtype02
ReadVariableOp_1┴
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_6_moving_mean*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOp╦
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╘
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3:А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3▒
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_6_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_6/moving_mean*
_output_shapes
 *
dtype02
AssignNewValue╟
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_6/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1М
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*'
_output_shapes
:А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:А::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:O K
'
_output_shapes
:А
 
_user_specified_nameinputs
Гч
ё!
__inference__wrapped_model_714
input_1;
7functional_1_conv2d_conv2d_readvariableop_conv2d_kernelM
Ifunctional_1_batch_normalization_readvariableop_batch_normalization_gammaN
Jfunctional_1_batch_normalization_readvariableop_1_batch_normalization_betad
`functional_1_batch_normalization_fusedbatchnormv3_readvariableop_batch_normalization_moving_meanj
ffunctional_1_batch_normalization_fusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance\
Xfunctional_1_depthwise_conv2d_depthwise_readvariableop_depthwise_conv2d_depthwise_kernelQ
Mfunctional_1_batch_normalization_1_readvariableop_batch_normalization_1_gammaR
Nfunctional_1_batch_normalization_1_readvariableop_1_batch_normalization_1_betah
dfunctional_1_batch_normalization_1_fusedbatchnormv3_readvariableop_batch_normalization_1_moving_meann
jfunctional_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance?
;functional_1_conv2d_1_conv2d_readvariableop_conv2d_1_kernelQ
Mfunctional_1_batch_normalization_2_readvariableop_batch_normalization_2_gammaR
Nfunctional_1_batch_normalization_2_readvariableop_1_batch_normalization_2_betah
dfunctional_1_batch_normalization_2_fusedbatchnormv3_readvariableop_batch_normalization_2_moving_meann
jfunctional_1_batch_normalization_2_fusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance`
\functional_1_depthwise_conv2d_1_depthwise_readvariableop_depthwise_conv2d_1_depthwise_kernelQ
Mfunctional_1_batch_normalization_3_readvariableop_batch_normalization_3_gammaR
Nfunctional_1_batch_normalization_3_readvariableop_1_batch_normalization_3_betah
dfunctional_1_batch_normalization_3_fusedbatchnormv3_readvariableop_batch_normalization_3_moving_meann
jfunctional_1_batch_normalization_3_fusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance?
;functional_1_conv2d_2_conv2d_readvariableop_conv2d_2_kernelQ
Mfunctional_1_batch_normalization_4_readvariableop_batch_normalization_4_gammaR
Nfunctional_1_batch_normalization_4_readvariableop_1_batch_normalization_4_betah
dfunctional_1_batch_normalization_4_fusedbatchnormv3_readvariableop_batch_normalization_4_moving_meann
jfunctional_1_batch_normalization_4_fusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance`
\functional_1_depthwise_conv2d_2_depthwise_readvariableop_depthwise_conv2d_2_depthwise_kernelQ
Mfunctional_1_batch_normalization_5_readvariableop_batch_normalization_5_gammaR
Nfunctional_1_batch_normalization_5_readvariableop_1_batch_normalization_5_betah
dfunctional_1_batch_normalization_5_fusedbatchnormv3_readvariableop_batch_normalization_5_moving_meann
jfunctional_1_batch_normalization_5_fusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance?
;functional_1_conv2d_3_conv2d_readvariableop_conv2d_3_kernelQ
Mfunctional_1_batch_normalization_6_readvariableop_batch_normalization_6_gammaR
Nfunctional_1_batch_normalization_6_readvariableop_1_batch_normalization_6_betah
dfunctional_1_batch_normalization_6_fusedbatchnormv3_readvariableop_batch_normalization_6_moving_meann
jfunctional_1_batch_normalization_6_fusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance`
\functional_1_depthwise_conv2d_3_depthwise_readvariableop_depthwise_conv2d_3_depthwise_kernelQ
Mfunctional_1_batch_normalization_7_readvariableop_batch_normalization_7_gammaR
Nfunctional_1_batch_normalization_7_readvariableop_1_batch_normalization_7_betah
dfunctional_1_batch_normalization_7_fusedbatchnormv3_readvariableop_batch_normalization_7_moving_meann
jfunctional_1_batch_normalization_7_fusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance?
;functional_1_conv2d_4_conv2d_readvariableop_conv2d_4_kernelQ
Mfunctional_1_batch_normalization_8_readvariableop_batch_normalization_8_gammaR
Nfunctional_1_batch_normalization_8_readvariableop_1_batch_normalization_8_betah
dfunctional_1_batch_normalization_8_fusedbatchnormv3_readvariableop_batch_normalization_8_moving_meann
jfunctional_1_batch_normalization_8_fusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance9
5functional_1_dense_matmul_readvariableop_dense_kernel8
4functional_1_dense_biasadd_readvariableop_dense_bias
identityИк
2functional_1/tf_op_layer_ExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :24
2functional_1/tf_op_layer_ExpandDims/ExpandDims/dimЇ
.functional_1/tf_op_layer_ExpandDims/ExpandDims
ExpandDimsinput_1;functional_1/tf_op_layer_ExpandDims/ExpandDims/dim:output:0*
T0*
_cloned(*&
_output_shapes
:1(20
.functional_1/tf_op_layer_ExpandDims/ExpandDims╓
)functional_1/conv2d/Conv2D/ReadVariableOpReadVariableOp7functional_1_conv2d_conv2d_readvariableop_conv2d_kernel*&
_output_shapes
:( *
dtype02+
)functional_1/conv2d/Conv2D/ReadVariableOpИ
functional_1/conv2d/Conv2DConv2D7functional_1/tf_op_layer_ExpandDims/ExpandDims:output:01functional_1/conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
2
functional_1/conv2d/Conv2Dш
/functional_1/batch_normalization/ReadVariableOpReadVariableOpIfunctional_1_batch_normalization_readvariableop_batch_normalization_gamma*
_output_shapes
: *
dtype021
/functional_1/batch_normalization/ReadVariableOpэ
1functional_1/batch_normalization/ReadVariableOp_1ReadVariableOpJfunctional_1_batch_normalization_readvariableop_1_batch_normalization_beta*
_output_shapes
: *
dtype023
1functional_1/batch_normalization/ReadVariableOp_1б
@functional_1/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp`functional_1_batch_normalization_fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
_output_shapes
: *
dtype02B
@functional_1/batch_normalization/FusedBatchNormV3/ReadVariableOpл
Bfunctional_1/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpffunctional_1_batch_normalization_fusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
_output_shapes
: *
dtype02D
Bfunctional_1/batch_normalization/FusedBatchNormV3/ReadVariableOp_1д
1functional_1/batch_normalization/FusedBatchNormV3FusedBatchNormV3#functional_1/conv2d/Conv2D:output:07functional_1/batch_normalization/ReadVariableOp:value:09functional_1/batch_normalization/ReadVariableOp_1:value:0Hfunctional_1/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Jfunctional_1/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%oГ:*
is_training( 23
1functional_1/batch_normalization/FusedBatchNormV3е
functional_1/re_lu/Relu6Relu65functional_1/batch_normalization/FusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2
functional_1/re_lu/Relu6С
6functional_1/depthwise_conv2d/depthwise/ReadVariableOpReadVariableOpXfunctional_1_depthwise_conv2d_depthwise_readvariableop_depthwise_conv2d_depthwise_kernel*&
_output_shapes
: *
dtype028
6functional_1/depthwise_conv2d/depthwise/ReadVariableOp╖
-functional_1/depthwise_conv2d/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-functional_1/depthwise_conv2d/depthwise/Shape┐
5functional_1/depthwise_conv2d/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      27
5functional_1/depthwise_conv2d/depthwise/dilation_rateн
'functional_1/depthwise_conv2d/depthwiseDepthwiseConv2dNative&functional_1/re_lu/Relu6:activations:0>functional_1/depthwise_conv2d/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
2)
'functional_1/depthwise_conv2d/depthwiseЁ
1functional_1/batch_normalization_1/ReadVariableOpReadVariableOpMfunctional_1_batch_normalization_1_readvariableop_batch_normalization_1_gamma*
_output_shapes
: *
dtype023
1functional_1/batch_normalization_1/ReadVariableOpї
3functional_1/batch_normalization_1/ReadVariableOp_1ReadVariableOpNfunctional_1_batch_normalization_1_readvariableop_1_batch_normalization_1_beta*
_output_shapes
: *
dtype025
3functional_1/batch_normalization_1/ReadVariableOp_1й
Bfunctional_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpdfunctional_1_batch_normalization_1_fusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
: *
dtype02D
Bfunctional_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp│
Dfunctional_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpjfunctional_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
_output_shapes
: *
dtype02F
Dfunctional_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1╜
3functional_1/batch_normalization_1/FusedBatchNormV3FusedBatchNormV30functional_1/depthwise_conv2d/depthwise:output:09functional_1/batch_normalization_1/ReadVariableOp:value:0;functional_1/batch_normalization_1/ReadVariableOp_1:value:0Jfunctional_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Lfunctional_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%oГ:*
is_training( 25
3functional_1/batch_normalization_1/FusedBatchNormV3л
functional_1/re_lu_1/Relu6Relu67functional_1/batch_normalization_1/FusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2
functional_1/re_lu_1/Relu6▐
+functional_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp;functional_1_conv2d_1_conv2d_readvariableop_conv2d_1_kernel*&
_output_shapes
:  *
dtype02-
+functional_1/conv2d_1/Conv2D/ReadVariableOp■
functional_1/conv2d_1/Conv2DConv2D(functional_1/re_lu_1/Relu6:activations:03functional_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingSAME*
strides
2
functional_1/conv2d_1/Conv2DЁ
1functional_1/batch_normalization_2/ReadVariableOpReadVariableOpMfunctional_1_batch_normalization_2_readvariableop_batch_normalization_2_gamma*
_output_shapes
: *
dtype023
1functional_1/batch_normalization_2/ReadVariableOpї
3functional_1/batch_normalization_2/ReadVariableOp_1ReadVariableOpNfunctional_1_batch_normalization_2_readvariableop_1_batch_normalization_2_beta*
_output_shapes
: *
dtype025
3functional_1/batch_normalization_2/ReadVariableOp_1й
Bfunctional_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpdfunctional_1_batch_normalization_2_fusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
: *
dtype02D
Bfunctional_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp│
Dfunctional_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpjfunctional_1_batch_normalization_2_fusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
_output_shapes
: *
dtype02F
Dfunctional_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1▓
3functional_1/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3%functional_1/conv2d_1/Conv2D:output:09functional_1/batch_normalization_2/ReadVariableOp:value:0;functional_1/batch_normalization_2/ReadVariableOp_1:value:0Jfunctional_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Lfunctional_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%oГ:*
is_training( 25
3functional_1/batch_normalization_2/FusedBatchNormV3л
functional_1/re_lu_2/Relu6Relu67functional_1/batch_normalization_2/FusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2
functional_1/re_lu_2/Relu6Щ
8functional_1/depthwise_conv2d_1/depthwise/ReadVariableOpReadVariableOp\functional_1_depthwise_conv2d_1_depthwise_readvariableop_depthwise_conv2d_1_depthwise_kernel*&
_output_shapes
: *
dtype02:
8functional_1/depthwise_conv2d_1/depthwise/ReadVariableOp╗
/functional_1/depthwise_conv2d_1/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             21
/functional_1/depthwise_conv2d_1/depthwise/Shape├
7functional_1/depthwise_conv2d_1/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      29
7functional_1/depthwise_conv2d_1/depthwise/dilation_rate╡
)functional_1/depthwise_conv2d_1/depthwiseDepthwiseConv2dNative(functional_1/re_lu_2/Relu6:activations:0@functional_1/depthwise_conv2d_1/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
2+
)functional_1/depthwise_conv2d_1/depthwiseЁ
1functional_1/batch_normalization_3/ReadVariableOpReadVariableOpMfunctional_1_batch_normalization_3_readvariableop_batch_normalization_3_gamma*
_output_shapes
: *
dtype023
1functional_1/batch_normalization_3/ReadVariableOpї
3functional_1/batch_normalization_3/ReadVariableOp_1ReadVariableOpNfunctional_1_batch_normalization_3_readvariableop_1_batch_normalization_3_beta*
_output_shapes
: *
dtype025
3functional_1/batch_normalization_3/ReadVariableOp_1й
Bfunctional_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpdfunctional_1_batch_normalization_3_fusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
_output_shapes
: *
dtype02D
Bfunctional_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp│
Dfunctional_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpjfunctional_1_batch_normalization_3_fusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
_output_shapes
: *
dtype02F
Dfunctional_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1┐
3functional_1/batch_normalization_3/FusedBatchNormV3FusedBatchNormV32functional_1/depthwise_conv2d_1/depthwise:output:09functional_1/batch_normalization_3/ReadVariableOp:value:0;functional_1/batch_normalization_3/ReadVariableOp_1:value:0Jfunctional_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Lfunctional_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%oГ:*
is_training( 25
3functional_1/batch_normalization_3/FusedBatchNormV3л
functional_1/re_lu_3/Relu6Relu67functional_1/batch_normalization_3/FusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2
functional_1/re_lu_3/Relu6▐
+functional_1/conv2d_2/Conv2D/ReadVariableOpReadVariableOp;functional_1_conv2d_2_conv2d_readvariableop_conv2d_2_kernel*&
_output_shapes
: @*
dtype02-
+functional_1/conv2d_2/Conv2D/ReadVariableOp■
functional_1/conv2d_2/Conv2DConv2D(functional_1/re_lu_3/Relu6:activations:03functional_1/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingSAME*
strides
2
functional_1/conv2d_2/Conv2DЁ
1functional_1/batch_normalization_4/ReadVariableOpReadVariableOpMfunctional_1_batch_normalization_4_readvariableop_batch_normalization_4_gamma*
_output_shapes
:@*
dtype023
1functional_1/batch_normalization_4/ReadVariableOpї
3functional_1/batch_normalization_4/ReadVariableOp_1ReadVariableOpNfunctional_1_batch_normalization_4_readvariableop_1_batch_normalization_4_beta*
_output_shapes
:@*
dtype025
3functional_1/batch_normalization_4/ReadVariableOp_1й
Bfunctional_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpdfunctional_1_batch_normalization_4_fusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
_output_shapes
:@*
dtype02D
Bfunctional_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp│
Dfunctional_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpjfunctional_1_batch_normalization_4_fusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
_output_shapes
:@*
dtype02F
Dfunctional_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1▓
3functional_1/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3%functional_1/conv2d_2/Conv2D:output:09functional_1/batch_normalization_4/ReadVariableOp:value:0;functional_1/batch_normalization_4/ReadVariableOp_1:value:0Jfunctional_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Lfunctional_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:@:@:@:@:@:*
epsilon%oГ:*
is_training( 25
3functional_1/batch_normalization_4/FusedBatchNormV3л
functional_1/re_lu_4/Relu6Relu67functional_1/batch_normalization_4/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:@2
functional_1/re_lu_4/Relu6Щ
8functional_1/depthwise_conv2d_2/depthwise/ReadVariableOpReadVariableOp\functional_1_depthwise_conv2d_2_depthwise_readvariableop_depthwise_conv2d_2_depthwise_kernel*&
_output_shapes
:@*
dtype02:
8functional_1/depthwise_conv2d_2/depthwise/ReadVariableOp╗
/functional_1/depthwise_conv2d_2/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @      21
/functional_1/depthwise_conv2d_2/depthwise/Shape├
7functional_1/depthwise_conv2d_2/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      29
7functional_1/depthwise_conv2d_2/depthwise/dilation_rate┤
)functional_1/depthwise_conv2d_2/depthwiseDepthwiseConv2dNative(functional_1/re_lu_4/Relu6:activations:0@functional_1/depthwise_conv2d_2/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingSAME*
strides
2+
)functional_1/depthwise_conv2d_2/depthwiseЁ
1functional_1/batch_normalization_5/ReadVariableOpReadVariableOpMfunctional_1_batch_normalization_5_readvariableop_batch_normalization_5_gamma*
_output_shapes
:@*
dtype023
1functional_1/batch_normalization_5/ReadVariableOpї
3functional_1/batch_normalization_5/ReadVariableOp_1ReadVariableOpNfunctional_1_batch_normalization_5_readvariableop_1_batch_normalization_5_beta*
_output_shapes
:@*
dtype025
3functional_1/batch_normalization_5/ReadVariableOp_1й
Bfunctional_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpdfunctional_1_batch_normalization_5_fusedbatchnormv3_readvariableop_batch_normalization_5_moving_mean*
_output_shapes
:@*
dtype02D
Bfunctional_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp│
Dfunctional_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpjfunctional_1_batch_normalization_5_fusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance*
_output_shapes
:@*
dtype02F
Dfunctional_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1┐
3functional_1/batch_normalization_5/FusedBatchNormV3FusedBatchNormV32functional_1/depthwise_conv2d_2/depthwise:output:09functional_1/batch_normalization_5/ReadVariableOp:value:0;functional_1/batch_normalization_5/ReadVariableOp_1:value:0Jfunctional_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Lfunctional_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:@:@:@:@:@:*
epsilon%oГ:*
is_training( 25
3functional_1/batch_normalization_5/FusedBatchNormV3л
functional_1/re_lu_5/Relu6Relu67functional_1/batch_normalization_5/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:@2
functional_1/re_lu_5/Relu6▀
+functional_1/conv2d_3/Conv2D/ReadVariableOpReadVariableOp;functional_1_conv2d_3_conv2d_readvariableop_conv2d_3_kernel*'
_output_shapes
:@А*
dtype02-
+functional_1/conv2d_3/Conv2D/ReadVariableOp 
functional_1/conv2d_3/Conv2DConv2D(functional_1/re_lu_5/Relu6:activations:03functional_1/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:А*
paddingSAME*
strides
2
functional_1/conv2d_3/Conv2Dё
1functional_1/batch_normalization_6/ReadVariableOpReadVariableOpMfunctional_1_batch_normalization_6_readvariableop_batch_normalization_6_gamma*
_output_shapes	
:А*
dtype023
1functional_1/batch_normalization_6/ReadVariableOpЎ
3functional_1/batch_normalization_6/ReadVariableOp_1ReadVariableOpNfunctional_1_batch_normalization_6_readvariableop_1_batch_normalization_6_beta*
_output_shapes	
:А*
dtype025
3functional_1/batch_normalization_6/ReadVariableOp_1к
Bfunctional_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpdfunctional_1_batch_normalization_6_fusedbatchnormv3_readvariableop_batch_normalization_6_moving_mean*
_output_shapes	
:А*
dtype02D
Bfunctional_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp┤
Dfunctional_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpjfunctional_1_batch_normalization_6_fusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance*
_output_shapes	
:А*
dtype02F
Dfunctional_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1╖
3functional_1/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3%functional_1/conv2d_3/Conv2D:output:09functional_1/batch_normalization_6/ReadVariableOp:value:0;functional_1/batch_normalization_6/ReadVariableOp_1:value:0Jfunctional_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Lfunctional_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3:А:А:А:А:А:*
epsilon%oГ:*
is_training( 25
3functional_1/batch_normalization_6/FusedBatchNormV3м
functional_1/re_lu_6/Relu6Relu67functional_1/batch_normalization_6/FusedBatchNormV3:y:0*
T0*'
_output_shapes
:А2
functional_1/re_lu_6/Relu6Ъ
8functional_1/depthwise_conv2d_3/depthwise/ReadVariableOpReadVariableOp\functional_1_depthwise_conv2d_3_depthwise_readvariableop_depthwise_conv2d_3_depthwise_kernel*'
_output_shapes
:А*
dtype02:
8functional_1/depthwise_conv2d_3/depthwise/ReadVariableOp╗
/functional_1/depthwise_conv2d_3/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      А      21
/functional_1/depthwise_conv2d_3/depthwise/Shape├
7functional_1/depthwise_conv2d_3/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      29
7functional_1/depthwise_conv2d_3/depthwise/dilation_rate╡
)functional_1/depthwise_conv2d_3/depthwiseDepthwiseConv2dNative(functional_1/re_lu_6/Relu6:activations:0@functional_1/depthwise_conv2d_3/depthwise/ReadVariableOp:value:0*
T0*'
_output_shapes
:А*
paddingSAME*
strides
2+
)functional_1/depthwise_conv2d_3/depthwiseё
1functional_1/batch_normalization_7/ReadVariableOpReadVariableOpMfunctional_1_batch_normalization_7_readvariableop_batch_normalization_7_gamma*
_output_shapes	
:А*
dtype023
1functional_1/batch_normalization_7/ReadVariableOpЎ
3functional_1/batch_normalization_7/ReadVariableOp_1ReadVariableOpNfunctional_1_batch_normalization_7_readvariableop_1_batch_normalization_7_beta*
_output_shapes	
:А*
dtype025
3functional_1/batch_normalization_7/ReadVariableOp_1к
Bfunctional_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpdfunctional_1_batch_normalization_7_fusedbatchnormv3_readvariableop_batch_normalization_7_moving_mean*
_output_shapes	
:А*
dtype02D
Bfunctional_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp┤
Dfunctional_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpjfunctional_1_batch_normalization_7_fusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance*
_output_shapes	
:А*
dtype02F
Dfunctional_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1─
3functional_1/batch_normalization_7/FusedBatchNormV3FusedBatchNormV32functional_1/depthwise_conv2d_3/depthwise:output:09functional_1/batch_normalization_7/ReadVariableOp:value:0;functional_1/batch_normalization_7/ReadVariableOp_1:value:0Jfunctional_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Lfunctional_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3:А:А:А:А:А:*
epsilon%oГ:*
is_training( 25
3functional_1/batch_normalization_7/FusedBatchNormV3м
functional_1/re_lu_7/Relu6Relu67functional_1/batch_normalization_7/FusedBatchNormV3:y:0*
T0*'
_output_shapes
:А2
functional_1/re_lu_7/Relu6р
+functional_1/conv2d_4/Conv2D/ReadVariableOpReadVariableOp;functional_1_conv2d_4_conv2d_readvariableop_conv2d_4_kernel*(
_output_shapes
:АА*
dtype02-
+functional_1/conv2d_4/Conv2D/ReadVariableOp 
functional_1/conv2d_4/Conv2DConv2D(functional_1/re_lu_7/Relu6:activations:03functional_1/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:А*
paddingSAME*
strides
2
functional_1/conv2d_4/Conv2Dё
1functional_1/batch_normalization_8/ReadVariableOpReadVariableOpMfunctional_1_batch_normalization_8_readvariableop_batch_normalization_8_gamma*
_output_shapes	
:А*
dtype023
1functional_1/batch_normalization_8/ReadVariableOpЎ
3functional_1/batch_normalization_8/ReadVariableOp_1ReadVariableOpNfunctional_1_batch_normalization_8_readvariableop_1_batch_normalization_8_beta*
_output_shapes	
:А*
dtype025
3functional_1/batch_normalization_8/ReadVariableOp_1к
Bfunctional_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpdfunctional_1_batch_normalization_8_fusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean*
_output_shapes	
:А*
dtype02D
Bfunctional_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp┤
Dfunctional_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpjfunctional_1_batch_normalization_8_fusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance*
_output_shapes	
:А*
dtype02F
Dfunctional_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1╖
3functional_1/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3%functional_1/conv2d_4/Conv2D:output:09functional_1/batch_normalization_8/ReadVariableOp:value:0;functional_1/batch_normalization_8/ReadVariableOp_1:value:0Jfunctional_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Lfunctional_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3:А:А:А:А:А:*
epsilon%oГ:*
is_training( 25
3functional_1/batch_normalization_8/FusedBatchNormV3м
functional_1/re_lu_8/Relu6Relu67functional_1/batch_normalization_8/FusedBatchNormV3:y:0*
T0*'
_output_shapes
:А2
functional_1/re_lu_8/Relu6═
<functional_1/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2>
<functional_1/global_average_pooling2d/Mean/reduction_indices√
*functional_1/global_average_pooling2d/MeanMean(functional_1/re_lu_8/Relu6:activations:0Efunctional_1/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*
_output_shapes
:	А2,
*functional_1/global_average_pooling2d/Meanй
functional_1/dropout/IdentityIdentity3functional_1/global_average_pooling2d/Mean:output:0*
T0*
_output_shapes
:	А2
functional_1/dropout/Identity╦
(functional_1/dense/MatMul/ReadVariableOpReadVariableOp5functional_1_dense_matmul_readvariableop_dense_kernel*
_output_shapes
:	А*
dtype02*
(functional_1/dense/MatMul/ReadVariableOp├
functional_1/dense/MatMulMatMul&functional_1/dropout/Identity:output:00functional_1/dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/dense/MatMul╟
)functional_1/dense/BiasAdd/ReadVariableOpReadVariableOp4functional_1_dense_biasadd_readvariableop_dense_bias*
_output_shapes
:*
dtype02+
)functional_1/dense/BiasAdd/ReadVariableOp─
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
identityIdentity:output:0*▀
_input_shapes═
╩:1(::::::::::::::::::::::::::::::::::::::::::::::::T P
+
_output_shapes
:         1(
!
_user_specified_name	input_1
▀
А
4__inference_batch_normalization_1_layer_call_fn_4062

inputs
batch_normalization_1_gamma
batch_normalization_1_beta%
!batch_normalization_1_moving_mean)
%batch_normalization_1_moving_variance
identityИвStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_1_gammabatch_normalization_1_beta!batch_normalization_1_moving_mean%batch_normalization_1_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_17982
StatefulPartitionedCallН
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": ::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
: 
 
_user_specified_nameinputs
шв
В#
F__inference_functional_1_layer_call_and_return_conditional_losses_3105
input_1.
*conv2d_conv2d_readvariableop_conv2d_kernel@
<batch_normalization_readvariableop_batch_normalization_gammaA
=batch_normalization_readvariableop_1_batch_normalization_betaW
Sbatch_normalization_fusedbatchnormv3_readvariableop_batch_normalization_moving_mean]
Ybatch_normalization_fusedbatchnormv3_readvariableop_1_batch_normalization_moving_varianceO
Kdepthwise_conv2d_depthwise_readvariableop_depthwise_conv2d_depthwise_kernelD
@batch_normalization_1_readvariableop_batch_normalization_1_gammaE
Abatch_normalization_1_readvariableop_1_batch_normalization_1_beta[
Wbatch_normalization_1_fusedbatchnormv3_readvariableop_batch_normalization_1_moving_meana
]batch_normalization_1_fusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance2
.conv2d_1_conv2d_readvariableop_conv2d_1_kernelD
@batch_normalization_2_readvariableop_batch_normalization_2_gammaE
Abatch_normalization_2_readvariableop_1_batch_normalization_2_beta[
Wbatch_normalization_2_fusedbatchnormv3_readvariableop_batch_normalization_2_moving_meana
]batch_normalization_2_fusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_varianceS
Odepthwise_conv2d_1_depthwise_readvariableop_depthwise_conv2d_1_depthwise_kernelD
@batch_normalization_3_readvariableop_batch_normalization_3_gammaE
Abatch_normalization_3_readvariableop_1_batch_normalization_3_beta[
Wbatch_normalization_3_fusedbatchnormv3_readvariableop_batch_normalization_3_moving_meana
]batch_normalization_3_fusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance2
.conv2d_2_conv2d_readvariableop_conv2d_2_kernelD
@batch_normalization_4_readvariableop_batch_normalization_4_gammaE
Abatch_normalization_4_readvariableop_1_batch_normalization_4_beta[
Wbatch_normalization_4_fusedbatchnormv3_readvariableop_batch_normalization_4_moving_meana
]batch_normalization_4_fusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_varianceS
Odepthwise_conv2d_2_depthwise_readvariableop_depthwise_conv2d_2_depthwise_kernelD
@batch_normalization_5_readvariableop_batch_normalization_5_gammaE
Abatch_normalization_5_readvariableop_1_batch_normalization_5_beta[
Wbatch_normalization_5_fusedbatchnormv3_readvariableop_batch_normalization_5_moving_meana
]batch_normalization_5_fusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance2
.conv2d_3_conv2d_readvariableop_conv2d_3_kernelD
@batch_normalization_6_readvariableop_batch_normalization_6_gammaE
Abatch_normalization_6_readvariableop_1_batch_normalization_6_beta[
Wbatch_normalization_6_fusedbatchnormv3_readvariableop_batch_normalization_6_moving_meana
]batch_normalization_6_fusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_varianceS
Odepthwise_conv2d_3_depthwise_readvariableop_depthwise_conv2d_3_depthwise_kernelD
@batch_normalization_7_readvariableop_batch_normalization_7_gammaE
Abatch_normalization_7_readvariableop_1_batch_normalization_7_beta[
Wbatch_normalization_7_fusedbatchnormv3_readvariableop_batch_normalization_7_moving_meana
]batch_normalization_7_fusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance2
.conv2d_4_conv2d_readvariableop_conv2d_4_kernelD
@batch_normalization_8_readvariableop_batch_normalization_8_gammaE
Abatch_normalization_8_readvariableop_1_batch_normalization_8_beta[
Wbatch_normalization_8_fusedbatchnormv3_readvariableop_batch_normalization_8_moving_meana
]batch_normalization_8_fusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance,
(dense_matmul_readvariableop_dense_kernel+
'dense_biasadd_readvariableop_dense_bias
identityИв"batch_normalization/AssignNewValueв$batch_normalization/AssignNewValue_1в$batch_normalization_1/AssignNewValueв&batch_normalization_1/AssignNewValue_1в$batch_normalization_2/AssignNewValueв&batch_normalization_2/AssignNewValue_1в$batch_normalization_3/AssignNewValueв&batch_normalization_3/AssignNewValue_1в$batch_normalization_4/AssignNewValueв&batch_normalization_4/AssignNewValue_1в$batch_normalization_5/AssignNewValueв&batch_normalization_5/AssignNewValue_1в$batch_normalization_6/AssignNewValueв&batch_normalization_6/AssignNewValue_1в$batch_normalization_7/AssignNewValueв&batch_normalization_7/AssignNewValue_1в$batch_normalization_8/AssignNewValueв&batch_normalization_8/AssignNewValue_1Р
%tf_op_layer_ExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%tf_op_layer_ExpandDims/ExpandDims/dim═
!tf_op_layer_ExpandDims/ExpandDims
ExpandDimsinput_1.tf_op_layer_ExpandDims/ExpandDims/dim:output:0*
T0*
_cloned(*&
_output_shapes
:1(2#
!tf_op_layer_ExpandDims/ExpandDimsп
conv2d/Conv2D/ReadVariableOpReadVariableOp*conv2d_conv2d_readvariableop_conv2d_kernel*&
_output_shapes
:( *
dtype02
conv2d/Conv2D/ReadVariableOp╘
conv2d/Conv2DConv2D*tf_op_layer_ExpandDims/ExpandDims:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
2
conv2d/Conv2D┴
"batch_normalization/ReadVariableOpReadVariableOp<batch_normalization_readvariableop_batch_normalization_gamma*
_output_shapes
: *
dtype02$
"batch_normalization/ReadVariableOp╞
$batch_normalization/ReadVariableOp_1ReadVariableOp=batch_normalization_readvariableop_1_batch_normalization_beta*
_output_shapes
: *
dtype02&
$batch_normalization/ReadVariableOp_1·
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpSbatch_normalization_fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
_output_shapes
: *
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpД
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpYbatch_normalization_fusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
_output_shapes
: *
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1╫
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/Conv2D:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2&
$batch_normalization/FusedBatchNormV3е
"batch_normalization/AssignNewValueAssignVariableOpSbatch_normalization_fusedbatchnormv3_readvariableop_batch_normalization_moving_mean1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*f
_class\
ZXloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp/batch_normalization/moving_mean*
_output_shapes
 *
dtype02$
"batch_normalization/AssignNewValue╗
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
re_lu/Relu6ъ
)depthwise_conv2d/depthwise/ReadVariableOpReadVariableOpKdepthwise_conv2d_depthwise_readvariableop_depthwise_conv2d_depthwise_kernel*&
_output_shapes
: *
dtype02+
)depthwise_conv2d/depthwise/ReadVariableOpЭ
 depthwise_conv2d/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 depthwise_conv2d/depthwise/Shapeе
(depthwise_conv2d/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2*
(depthwise_conv2d/depthwise/dilation_rate∙
depthwise_conv2d/depthwiseDepthwiseConv2dNativere_lu/Relu6:activations:01depthwise_conv2d/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
2
depthwise_conv2d/depthwise╔
$batch_normalization_1/ReadVariableOpReadVariableOp@batch_normalization_1_readvariableop_batch_normalization_1_gamma*
_output_shapes
: *
dtype02&
$batch_normalization_1/ReadVariableOp╬
&batch_normalization_1/ReadVariableOp_1ReadVariableOpAbatch_normalization_1_readvariableop_1_batch_normalization_1_beta*
_output_shapes
: *
dtype02(
&batch_normalization_1/ReadVariableOp_1В
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_1_fusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
: *
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpМ
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_1_fusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
_output_shapes
: *
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Ё
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3#depthwise_conv2d/depthwise:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2(
&batch_normalization_1/FusedBatchNormV3╡
$batch_normalization_1/AssignNewValueAssignVariableOpWbatch_normalization_1_fusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*j
_class`
^\loc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp/batch_normalization_1/moving_mean*
_output_shapes
 *
dtype02&
$batch_normalization_1/AssignNewValue╦
&batch_normalization_1/AssignNewValue_1AssignVariableOp]batch_normalization_1_fusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*p
_classf
dbloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_1/moving_variance*
_output_shapes
 *
dtype02(
&batch_normalization_1/AssignNewValue_1Д
re_lu_1/Relu6Relu6*batch_normalization_1/FusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2
re_lu_1/Relu6╖
conv2d_1/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_conv2d_readvariableop_conv2d_1_kernel*&
_output_shapes
:  *
dtype02 
conv2d_1/Conv2D/ReadVariableOp╩
conv2d_1/Conv2DConv2Dre_lu_1/Relu6:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingSAME*
strides
2
conv2d_1/Conv2D╔
$batch_normalization_2/ReadVariableOpReadVariableOp@batch_normalization_2_readvariableop_batch_normalization_2_gamma*
_output_shapes
: *
dtype02&
$batch_normalization_2/ReadVariableOp╬
&batch_normalization_2/ReadVariableOp_1ReadVariableOpAbatch_normalization_2_readvariableop_1_batch_normalization_2_beta*
_output_shapes
: *
dtype02(
&batch_normalization_2/ReadVariableOp_1В
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_2_fusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
: *
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpМ
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_2_fusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
_output_shapes
: *
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1х
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_1/Conv2D:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2(
&batch_normalization_2/FusedBatchNormV3╡
$batch_normalization_2/AssignNewValueAssignVariableOpWbatch_normalization_2_fusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*j
_class`
^\loc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp/batch_normalization_2/moving_mean*
_output_shapes
 *
dtype02&
$batch_normalization_2/AssignNewValue╦
&batch_normalization_2/AssignNewValue_1AssignVariableOp]batch_normalization_2_fusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*p
_classf
dbloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_2/moving_variance*
_output_shapes
 *
dtype02(
&batch_normalization_2/AssignNewValue_1Д
re_lu_2/Relu6Relu6*batch_normalization_2/FusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2
re_lu_2/Relu6Є
+depthwise_conv2d_1/depthwise/ReadVariableOpReadVariableOpOdepthwise_conv2d_1_depthwise_readvariableop_depthwise_conv2d_1_depthwise_kernel*&
_output_shapes
: *
dtype02-
+depthwise_conv2d_1/depthwise/ReadVariableOpб
"depthwise_conv2d_1/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"depthwise_conv2d_1/depthwise/Shapeй
*depthwise_conv2d_1/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2,
*depthwise_conv2d_1/depthwise/dilation_rateБ
depthwise_conv2d_1/depthwiseDepthwiseConv2dNativere_lu_2/Relu6:activations:03depthwise_conv2d_1/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
2
depthwise_conv2d_1/depthwise╔
$batch_normalization_3/ReadVariableOpReadVariableOp@batch_normalization_3_readvariableop_batch_normalization_3_gamma*
_output_shapes
: *
dtype02&
$batch_normalization_3/ReadVariableOp╬
&batch_normalization_3/ReadVariableOp_1ReadVariableOpAbatch_normalization_3_readvariableop_1_batch_normalization_3_beta*
_output_shapes
: *
dtype02(
&batch_normalization_3/ReadVariableOp_1В
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_3_fusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
_output_shapes
: *
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpМ
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_3_fusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
_output_shapes
: *
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Є
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3%depthwise_conv2d_1/depthwise:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2(
&batch_normalization_3/FusedBatchNormV3╡
$batch_normalization_3/AssignNewValueAssignVariableOpWbatch_normalization_3_fusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*j
_class`
^\loc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp/batch_normalization_3/moving_mean*
_output_shapes
 *
dtype02&
$batch_normalization_3/AssignNewValue╦
&batch_normalization_3/AssignNewValue_1AssignVariableOp]batch_normalization_3_fusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance7batch_normalization_3/FusedBatchNormV3:batch_variance:08^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*p
_classf
dbloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_3/moving_variance*
_output_shapes
 *
dtype02(
&batch_normalization_3/AssignNewValue_1Д
re_lu_3/Relu6Relu6*batch_normalization_3/FusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2
re_lu_3/Relu6╖
conv2d_2/Conv2D/ReadVariableOpReadVariableOp.conv2d_2_conv2d_readvariableop_conv2d_2_kernel*&
_output_shapes
: @*
dtype02 
conv2d_2/Conv2D/ReadVariableOp╩
conv2d_2/Conv2DConv2Dre_lu_3/Relu6:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingSAME*
strides
2
conv2d_2/Conv2D╔
$batch_normalization_4/ReadVariableOpReadVariableOp@batch_normalization_4_readvariableop_batch_normalization_4_gamma*
_output_shapes
:@*
dtype02&
$batch_normalization_4/ReadVariableOp╬
&batch_normalization_4/ReadVariableOp_1ReadVariableOpAbatch_normalization_4_readvariableop_1_batch_normalization_4_beta*
_output_shapes
:@*
dtype02(
&batch_normalization_4/ReadVariableOp_1В
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_4_fusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
_output_shapes
:@*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpМ
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_4_fusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
_output_shapes
:@*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1х
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_2/Conv2D:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2(
&batch_normalization_4/FusedBatchNormV3╡
$batch_normalization_4/AssignNewValueAssignVariableOpWbatch_normalization_4_fusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean3batch_normalization_4/FusedBatchNormV3:batch_mean:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp*j
_class`
^\loc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp/batch_normalization_4/moving_mean*
_output_shapes
 *
dtype02&
$batch_normalization_4/AssignNewValue╦
&batch_normalization_4/AssignNewValue_1AssignVariableOp]batch_normalization_4_fusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance7batch_normalization_4/FusedBatchNormV3:batch_variance:08^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*p
_classf
dbloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_4/moving_variance*
_output_shapes
 *
dtype02(
&batch_normalization_4/AssignNewValue_1Д
re_lu_4/Relu6Relu6*batch_normalization_4/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:@2
re_lu_4/Relu6Є
+depthwise_conv2d_2/depthwise/ReadVariableOpReadVariableOpOdepthwise_conv2d_2_depthwise_readvariableop_depthwise_conv2d_2_depthwise_kernel*&
_output_shapes
:@*
dtype02-
+depthwise_conv2d_2/depthwise/ReadVariableOpб
"depthwise_conv2d_2/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @      2$
"depthwise_conv2d_2/depthwise/Shapeй
*depthwise_conv2d_2/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2,
*depthwise_conv2d_2/depthwise/dilation_rateА
depthwise_conv2d_2/depthwiseDepthwiseConv2dNativere_lu_4/Relu6:activations:03depthwise_conv2d_2/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingSAME*
strides
2
depthwise_conv2d_2/depthwise╔
$batch_normalization_5/ReadVariableOpReadVariableOp@batch_normalization_5_readvariableop_batch_normalization_5_gamma*
_output_shapes
:@*
dtype02&
$batch_normalization_5/ReadVariableOp╬
&batch_normalization_5/ReadVariableOp_1ReadVariableOpAbatch_normalization_5_readvariableop_1_batch_normalization_5_beta*
_output_shapes
:@*
dtype02(
&batch_normalization_5/ReadVariableOp_1В
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_5_fusedbatchnormv3_readvariableop_batch_normalization_5_moving_mean*
_output_shapes
:@*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpМ
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_5_fusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance*
_output_shapes
:@*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Є
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3%depthwise_conv2d_2/depthwise:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2(
&batch_normalization_5/FusedBatchNormV3╡
$batch_normalization_5/AssignNewValueAssignVariableOpWbatch_normalization_5_fusedbatchnormv3_readvariableop_batch_normalization_5_moving_mean3batch_normalization_5/FusedBatchNormV3:batch_mean:06^batch_normalization_5/FusedBatchNormV3/ReadVariableOp*j
_class`
^\loc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp/batch_normalization_5/moving_mean*
_output_shapes
 *
dtype02&
$batch_normalization_5/AssignNewValue╦
&batch_normalization_5/AssignNewValue_1AssignVariableOp]batch_normalization_5_fusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance7batch_normalization_5/FusedBatchNormV3:batch_variance:08^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*p
_classf
dbloc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_5/moving_variance*
_output_shapes
 *
dtype02(
&batch_normalization_5/AssignNewValue_1Д
re_lu_5/Relu6Relu6*batch_normalization_5/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:@2
re_lu_5/Relu6╕
conv2d_3/Conv2D/ReadVariableOpReadVariableOp.conv2d_3_conv2d_readvariableop_conv2d_3_kernel*'
_output_shapes
:@А*
dtype02 
conv2d_3/Conv2D/ReadVariableOp╦
conv2d_3/Conv2DConv2Dre_lu_5/Relu6:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:А*
paddingSAME*
strides
2
conv2d_3/Conv2D╩
$batch_normalization_6/ReadVariableOpReadVariableOp@batch_normalization_6_readvariableop_batch_normalization_6_gamma*
_output_shapes	
:А*
dtype02&
$batch_normalization_6/ReadVariableOp╧
&batch_normalization_6/ReadVariableOp_1ReadVariableOpAbatch_normalization_6_readvariableop_1_batch_normalization_6_beta*
_output_shapes	
:А*
dtype02(
&batch_normalization_6/ReadVariableOp_1Г
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_6_fusedbatchnormv3_readvariableop_batch_normalization_6_moving_mean*
_output_shapes	
:А*
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpН
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_6_fusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance*
_output_shapes	
:А*
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ъ
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv2d_3/Conv2D:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3:А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2(
&batch_normalization_6/FusedBatchNormV3╡
$batch_normalization_6/AssignNewValueAssignVariableOpWbatch_normalization_6_fusedbatchnormv3_readvariableop_batch_normalization_6_moving_mean3batch_normalization_6/FusedBatchNormV3:batch_mean:06^batch_normalization_6/FusedBatchNormV3/ReadVariableOp*j
_class`
^\loc:@batch_normalization_6/FusedBatchNormV3/ReadVariableOp/batch_normalization_6/moving_mean*
_output_shapes
 *
dtype02&
$batch_normalization_6/AssignNewValue╦
&batch_normalization_6/AssignNewValue_1AssignVariableOp]batch_normalization_6_fusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance7batch_normalization_6/FusedBatchNormV3:batch_variance:08^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1*p
_classf
dbloc:@batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_6/moving_variance*
_output_shapes
 *
dtype02(
&batch_normalization_6/AssignNewValue_1Е
re_lu_6/Relu6Relu6*batch_normalization_6/FusedBatchNormV3:y:0*
T0*'
_output_shapes
:А2
re_lu_6/Relu6є
+depthwise_conv2d_3/depthwise/ReadVariableOpReadVariableOpOdepthwise_conv2d_3_depthwise_readvariableop_depthwise_conv2d_3_depthwise_kernel*'
_output_shapes
:А*
dtype02-
+depthwise_conv2d_3/depthwise/ReadVariableOpб
"depthwise_conv2d_3/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      А      2$
"depthwise_conv2d_3/depthwise/Shapeй
*depthwise_conv2d_3/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2,
*depthwise_conv2d_3/depthwise/dilation_rateБ
depthwise_conv2d_3/depthwiseDepthwiseConv2dNativere_lu_6/Relu6:activations:03depthwise_conv2d_3/depthwise/ReadVariableOp:value:0*
T0*'
_output_shapes
:А*
paddingSAME*
strides
2
depthwise_conv2d_3/depthwise╩
$batch_normalization_7/ReadVariableOpReadVariableOp@batch_normalization_7_readvariableop_batch_normalization_7_gamma*
_output_shapes	
:А*
dtype02&
$batch_normalization_7/ReadVariableOp╧
&batch_normalization_7/ReadVariableOp_1ReadVariableOpAbatch_normalization_7_readvariableop_1_batch_normalization_7_beta*
_output_shapes	
:А*
dtype02(
&batch_normalization_7/ReadVariableOp_1Г
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_7_fusedbatchnormv3_readvariableop_batch_normalization_7_moving_mean*
_output_shapes	
:А*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpН
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_7_fusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance*
_output_shapes	
:А*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ў
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3%depthwise_conv2d_3/depthwise:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3:А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2(
&batch_normalization_7/FusedBatchNormV3╡
$batch_normalization_7/AssignNewValueAssignVariableOpWbatch_normalization_7_fusedbatchnormv3_readvariableop_batch_normalization_7_moving_mean3batch_normalization_7/FusedBatchNormV3:batch_mean:06^batch_normalization_7/FusedBatchNormV3/ReadVariableOp*j
_class`
^\loc:@batch_normalization_7/FusedBatchNormV3/ReadVariableOp/batch_normalization_7/moving_mean*
_output_shapes
 *
dtype02&
$batch_normalization_7/AssignNewValue╦
&batch_normalization_7/AssignNewValue_1AssignVariableOp]batch_normalization_7_fusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance7batch_normalization_7/FusedBatchNormV3:batch_variance:08^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1*p
_classf
dbloc:@batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_7/moving_variance*
_output_shapes
 *
dtype02(
&batch_normalization_7/AssignNewValue_1Е
re_lu_7/Relu6Relu6*batch_normalization_7/FusedBatchNormV3:y:0*
T0*'
_output_shapes
:А2
re_lu_7/Relu6╣
conv2d_4/Conv2D/ReadVariableOpReadVariableOp.conv2d_4_conv2d_readvariableop_conv2d_4_kernel*(
_output_shapes
:АА*
dtype02 
conv2d_4/Conv2D/ReadVariableOp╦
conv2d_4/Conv2DConv2Dre_lu_7/Relu6:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:А*
paddingSAME*
strides
2
conv2d_4/Conv2D╩
$batch_normalization_8/ReadVariableOpReadVariableOp@batch_normalization_8_readvariableop_batch_normalization_8_gamma*
_output_shapes	
:А*
dtype02&
$batch_normalization_8/ReadVariableOp╧
&batch_normalization_8/ReadVariableOp_1ReadVariableOpAbatch_normalization_8_readvariableop_1_batch_normalization_8_beta*
_output_shapes	
:А*
dtype02(
&batch_normalization_8/ReadVariableOp_1Г
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_8_fusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean*
_output_shapes	
:А*
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpН
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_8_fusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance*
_output_shapes	
:А*
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ъ
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3conv2d_4/Conv2D:output:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3:А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2(
&batch_normalization_8/FusedBatchNormV3╡
$batch_normalization_8/AssignNewValueAssignVariableOpWbatch_normalization_8_fusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean3batch_normalization_8/FusedBatchNormV3:batch_mean:06^batch_normalization_8/FusedBatchNormV3/ReadVariableOp*j
_class`
^\loc:@batch_normalization_8/FusedBatchNormV3/ReadVariableOp/batch_normalization_8/moving_mean*
_output_shapes
 *
dtype02&
$batch_normalization_8/AssignNewValue╦
&batch_normalization_8/AssignNewValue_1AssignVariableOp]batch_normalization_8_fusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance7batch_normalization_8/FusedBatchNormV3:batch_variance:08^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1*p
_classf
dbloc:@batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_8/moving_variance*
_output_shapes
 *
dtype02(
&batch_normalization_8/AssignNewValue_1Е
re_lu_8/Relu6Relu6*batch_normalization_8/FusedBatchNormV3:y:0*
T0*'
_output_shapes
:А2
re_lu_8/Relu6│
/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      21
/global_average_pooling2d/Mean/reduction_indices╟
global_average_pooling2d/MeanMeanre_lu_8/Relu6:activations:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*
_output_shapes
:	А2
global_average_pooling2d/Means
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/dropout/Constг
dropout/dropout/MulMul&global_average_pooling2d/Mean:output:0dropout/dropout/Const:output:0*
T0*
_output_shapes
:	А2
dropout/dropout/Mul
dropout/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   А   2
dropout/dropout/Shape─
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*
_output_shapes
:	А*
dtype02.
,dropout/dropout/random_uniform/RandomUniformЕ
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2 
dropout/dropout/GreaterEqual/y╓
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	А2
dropout/dropout/GreaterEqualП
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	А2
dropout/dropout/CastТ
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*
_output_shapes
:	А2
dropout/dropout/Mul_1д
dense/MatMul/ReadVariableOpReadVariableOp(dense_matmul_readvariableop_dense_kernel*
_output_shapes
:	А*
dtype02
dense/MatMul/ReadVariableOpП
dense/MatMulMatMuldropout/dropout/Mul_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense/MatMulа
dense/BiasAdd/ReadVariableOpReadVariableOp'dense_biasadd_readvariableop_dense_bias*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOpР
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense/BiasAddн
IdentityIdentitydense/BiasAdd:output:0#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_1%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_1%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_1%^batch_normalization_4/AssignNewValue'^batch_normalization_4/AssignNewValue_1%^batch_normalization_5/AssignNewValue'^batch_normalization_5/AssignNewValue_1%^batch_normalization_6/AssignNewValue'^batch_normalization_6/AssignNewValue_1%^batch_normalization_7/AssignNewValue'^batch_normalization_7/AssignNewValue_1%^batch_normalization_8/AssignNewValue'^batch_normalization_8/AssignNewValue_1*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*▀
_input_shapes═
╩:1(:::::::::::::::::::::::::::::::::::::::::::::::2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12L
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
&batch_normalization_8/AssignNewValue_1&batch_normalization_8/AssignNewValue_1:T P
+
_output_shapes
:         1(
!
_user_specified_name	input_1
 
З
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1887

inputs.
*readvariableop_batch_normalization_2_gamma/
+readvariableop_1_batch_normalization_2_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_2_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance
identityИвAssignNewValueвAssignNewValue_1З
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_2_gamma*
_output_shapes
: *
dtype02
ReadVariableOpМ
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_2_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1└
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp╩
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╧
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3▒
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_2_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_2/moving_mean*
_output_shapes
 *
dtype02
AssignNewValue╟
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_2/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1Л
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:N J
&
_output_shapes
: 
 
_user_specified_nameinputs
╥
l
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_3881

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dimЗ

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
й
н
?__inference_dense_layer_call_and_return_conditional_losses_5050

inputs&
"matmul_readvariableop_dense_kernel%
!biasadd_readvariableop_dense_bias
identityИТ
MatMul/ReadVariableOpReadVariableOp"matmul_readvariableop_dense_kernel*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOpj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
MatMulО
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
identityIdentity:output:0*&
_input_shapes
:	А:::G C

_output_shapes
:	А
 
_user_specified_nameinputs
ы
З
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1192

inputs.
*readvariableop_batch_normalization_4_gamma/
+readvariableop_1_batch_normalization_4_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_4_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance
identityИвAssignNewValueвAssignNewValue_1З
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_4_gamma*
_output_shapes
:@*
dtype02
ReadVariableOpМ
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_4_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1└
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp╩
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3▒
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_4_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_4/moving_mean*
_output_shapes
 *
dtype02
AssignNewValue╟
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_4/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1ж
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
п
]
A__inference_re_lu_4_layer_call_and_return_conditional_losses_4510

inputs
identityP
Relu6Relu6inputs*
T0*&
_output_shapes
:@2
Relu6f
IdentityIdentityRelu6:activations:0*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*%
_input_shapes
:@:N J
&
_output_shapes
:@
 
_user_specified_nameinputs
 
З
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1967

inputs.
*readvariableop_batch_normalization_3_gamma/
+readvariableop_1_batch_normalization_3_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_3_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance
identityИвAssignNewValueвAssignNewValue_1З
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_3_gamma*
_output_shapes
: *
dtype02
ReadVariableOpМ
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_3_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1└
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp╩
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╧
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3▒
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_3_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_3/moving_mean*
_output_shapes
 *
dtype02
AssignNewValue╟
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_3/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1Л
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:N J
&
_output_shapes
: 
 
_user_specified_nameinputs
 
З
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2056

inputs.
*readvariableop_batch_normalization_4_gamma/
+readvariableop_1_batch_normalization_4_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_4_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance
identityИвAssignNewValueвAssignNewValue_1З
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_4_gamma*
_output_shapes
:@*
dtype02
ReadVariableOpМ
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_4_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1└
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp╩
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╧
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3▒
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_4_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_4/moving_mean*
_output_shapes
 *
dtype02
AssignNewValue╟
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_4/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1Л
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*5
_input_shapes$
":@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:N J
&
_output_shapes
:@
 
_user_specified_nameinputs
─
И
@__inference_conv2d_layer_call_and_return_conditional_losses_3893

inputs'
#conv2d_readvariableop_conv2d_kernel
identityИЪ
Conv2D/ReadVariableOpReadVariableOp#conv2d_readvariableop_conv2d_kernel*&
_output_shapes
:( *
dtype02
Conv2D/ReadVariableOpЫ
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
п
]
A__inference_re_lu_3_layer_call_and_return_conditional_losses_4379

inputs
identityP
Relu6Relu6inputs*
T0*&
_output_shapes
: 2
Relu6f
IdentityIdentityRelu6:activations:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*%
_input_shapes
: :N J
&
_output_shapes
: 
 
_user_specified_nameinputs
█
у
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_4985

inputs.
*readvariableop_batch_normalization_8_gamma/
+readvariableop_1_batch_normalization_8_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_8_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance
identityИИ
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_8_gamma*
_output_shapes	
:А*
dtype02
ReadVariableOpН
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_8_beta*
_output_shapes	
:А*
dtype02
ReadVariableOp_1┴
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOp╦
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1с
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Г
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,                           А:::::j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
╝╩
│
 __inference__traced_restore_5372
file_prefix"
assignvariableop_conv2d_kernel0
,assignvariableop_1_batch_normalization_gamma/
+assignvariableop_2_batch_normalization_beta6
2assignvariableop_3_batch_normalization_moving_mean:
6assignvariableop_4_batch_normalization_moving_variance8
4assignvariableop_5_depthwise_conv2d_depthwise_kernel2
.assignvariableop_6_batch_normalization_1_gamma1
-assignvariableop_7_batch_normalization_1_beta8
4assignvariableop_8_batch_normalization_1_moving_mean<
8assignvariableop_9_batch_normalization_1_moving_variance'
#assignvariableop_10_conv2d_1_kernel3
/assignvariableop_11_batch_normalization_2_gamma2
.assignvariableop_12_batch_normalization_2_beta9
5assignvariableop_13_batch_normalization_2_moving_mean=
9assignvariableop_14_batch_normalization_2_moving_variance;
7assignvariableop_15_depthwise_conv2d_1_depthwise_kernel3
/assignvariableop_16_batch_normalization_3_gamma2
.assignvariableop_17_batch_normalization_3_beta9
5assignvariableop_18_batch_normalization_3_moving_mean=
9assignvariableop_19_batch_normalization_3_moving_variance'
#assignvariableop_20_conv2d_2_kernel3
/assignvariableop_21_batch_normalization_4_gamma2
.assignvariableop_22_batch_normalization_4_beta9
5assignvariableop_23_batch_normalization_4_moving_mean=
9assignvariableop_24_batch_normalization_4_moving_variance;
7assignvariableop_25_depthwise_conv2d_2_depthwise_kernel3
/assignvariableop_26_batch_normalization_5_gamma2
.assignvariableop_27_batch_normalization_5_beta9
5assignvariableop_28_batch_normalization_5_moving_mean=
9assignvariableop_29_batch_normalization_5_moving_variance'
#assignvariableop_30_conv2d_3_kernel3
/assignvariableop_31_batch_normalization_6_gamma2
.assignvariableop_32_batch_normalization_6_beta9
5assignvariableop_33_batch_normalization_6_moving_mean=
9assignvariableop_34_batch_normalization_6_moving_variance;
7assignvariableop_35_depthwise_conv2d_3_depthwise_kernel3
/assignvariableop_36_batch_normalization_7_gamma2
.assignvariableop_37_batch_normalization_7_beta9
5assignvariableop_38_batch_normalization_7_moving_mean=
9assignvariableop_39_batch_normalization_7_moving_variance'
#assignvariableop_40_conv2d_4_kernel3
/assignvariableop_41_batch_normalization_8_gamma2
.assignvariableop_42_batch_normalization_8_beta9
5assignvariableop_43_batch_normalization_8_moving_mean=
9assignvariableop_44_batch_normalization_8_moving_variance$
 assignvariableop_45_dense_kernel"
assignvariableop_46_dense_bias
identity_48ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_37вAssignVariableOp_38вAssignVariableOp_39вAssignVariableOp_4вAssignVariableOp_40вAssignVariableOp_41вAssignVariableOp_42вAssignVariableOp_43вAssignVariableOp_44вAssignVariableOp_45вAssignVariableOp_46вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9О
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*Ъ
valueРBН0B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-6/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-10/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-14/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-17/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-17/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-17/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesю
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*s
valuejBh0B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesЮ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*╓
_output_shapes├
└::::::::::::::::::::::::::::::::::::::::::::::::*>
dtypes4
2202
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЭ
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1▒
AssignVariableOp_1AssignVariableOp,assignvariableop_1_batch_normalization_gammaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2░
AssignVariableOp_2AssignVariableOp+assignvariableop_2_batch_normalization_betaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3╖
AssignVariableOp_3AssignVariableOp2assignvariableop_3_batch_normalization_moving_meanIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4╗
AssignVariableOp_4AssignVariableOp6assignvariableop_4_batch_normalization_moving_varianceIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5╣
AssignVariableOp_5AssignVariableOp4assignvariableop_5_depthwise_conv2d_depthwise_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6│
AssignVariableOp_6AssignVariableOp.assignvariableop_6_batch_normalization_1_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7▓
AssignVariableOp_7AssignVariableOp-assignvariableop_7_batch_normalization_1_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8╣
AssignVariableOp_8AssignVariableOp4assignvariableop_8_batch_normalization_1_moving_meanIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9╜
AssignVariableOp_9AssignVariableOp8assignvariableop_9_batch_normalization_1_moving_varianceIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10л
AssignVariableOp_10AssignVariableOp#assignvariableop_10_conv2d_1_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11╖
AssignVariableOp_11AssignVariableOp/assignvariableop_11_batch_normalization_2_gammaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12╢
AssignVariableOp_12AssignVariableOp.assignvariableop_12_batch_normalization_2_betaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13╜
AssignVariableOp_13AssignVariableOp5assignvariableop_13_batch_normalization_2_moving_meanIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14┴
AssignVariableOp_14AssignVariableOp9assignvariableop_14_batch_normalization_2_moving_varianceIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15┐
AssignVariableOp_15AssignVariableOp7assignvariableop_15_depthwise_conv2d_1_depthwise_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16╖
AssignVariableOp_16AssignVariableOp/assignvariableop_16_batch_normalization_3_gammaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17╢
AssignVariableOp_17AssignVariableOp.assignvariableop_17_batch_normalization_3_betaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18╜
AssignVariableOp_18AssignVariableOp5assignvariableop_18_batch_normalization_3_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19┴
AssignVariableOp_19AssignVariableOp9assignvariableop_19_batch_normalization_3_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20л
AssignVariableOp_20AssignVariableOp#assignvariableop_20_conv2d_2_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21╖
AssignVariableOp_21AssignVariableOp/assignvariableop_21_batch_normalization_4_gammaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22╢
AssignVariableOp_22AssignVariableOp.assignvariableop_22_batch_normalization_4_betaIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23╜
AssignVariableOp_23AssignVariableOp5assignvariableop_23_batch_normalization_4_moving_meanIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24┴
AssignVariableOp_24AssignVariableOp9assignvariableop_24_batch_normalization_4_moving_varianceIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25┐
AssignVariableOp_25AssignVariableOp7assignvariableop_25_depthwise_conv2d_2_depthwise_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26╖
AssignVariableOp_26AssignVariableOp/assignvariableop_26_batch_normalization_5_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27╢
AssignVariableOp_27AssignVariableOp.assignvariableop_27_batch_normalization_5_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28╜
AssignVariableOp_28AssignVariableOp5assignvariableop_28_batch_normalization_5_moving_meanIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29┴
AssignVariableOp_29AssignVariableOp9assignvariableop_29_batch_normalization_5_moving_varianceIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30л
AssignVariableOp_30AssignVariableOp#assignvariableop_30_conv2d_3_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31╖
AssignVariableOp_31AssignVariableOp/assignvariableop_31_batch_normalization_6_gammaIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32╢
AssignVariableOp_32AssignVariableOp.assignvariableop_32_batch_normalization_6_betaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33╜
AssignVariableOp_33AssignVariableOp5assignvariableop_33_batch_normalization_6_moving_meanIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34┴
AssignVariableOp_34AssignVariableOp9assignvariableop_34_batch_normalization_6_moving_varianceIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35┐
AssignVariableOp_35AssignVariableOp7assignvariableop_35_depthwise_conv2d_3_depthwise_kernelIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36╖
AssignVariableOp_36AssignVariableOp/assignvariableop_36_batch_normalization_7_gammaIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37╢
AssignVariableOp_37AssignVariableOp.assignvariableop_37_batch_normalization_7_betaIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38╜
AssignVariableOp_38AssignVariableOp5assignvariableop_38_batch_normalization_7_moving_meanIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39┴
AssignVariableOp_39AssignVariableOp9assignvariableop_39_batch_normalization_7_moving_varianceIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40л
AssignVariableOp_40AssignVariableOp#assignvariableop_40_conv2d_4_kernelIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41╖
AssignVariableOp_41AssignVariableOp/assignvariableop_41_batch_normalization_8_gammaIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42╢
AssignVariableOp_42AssignVariableOp.assignvariableop_42_batch_normalization_8_betaIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43╜
AssignVariableOp_43AssignVariableOp5assignvariableop_43_batch_normalization_8_moving_meanIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44┴
AssignVariableOp_44AssignVariableOp9assignvariableop_44_batch_normalization_8_moving_varianceIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45и
AssignVariableOp_45AssignVariableOp assignvariableop_45_dense_kernelIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46ж
AssignVariableOp_46AssignVariableOpassignvariableop_46_dense_biasIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_469
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpш
Identity_47Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_47█
Identity_48IdentityIdentity_47:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_48"#
identity_48Identity_48:output:0*╙
_input_shapes┴
╛: :::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_46AssignVariableOp_462(
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
╧
у
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_4238

inputs.
*readvariableop_batch_normalization_2_gamma/
+readvariableop_1_batch_normalization_2_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_2_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance
identityИЗ
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_2_gamma*
_output_shapes
: *
dtype02
ReadVariableOpМ
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_2_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1└
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp╩
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            :::::i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
╧
у
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1219

inputs.
*readvariableop_batch_normalization_4_gamma/
+readvariableop_1_batch_normalization_4_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_4_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance
identityИЗ
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_4_gamma*
_output_shapes
:@*
dtype02
ReadVariableOpМ
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_4_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1└
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp╩
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           @:::::i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
╠
М
B__inference_conv2d_3_layer_call_and_return_conditional_losses_4640

inputs)
%conv2d_readvariableop_conv2d_3_kernel
identityИЭ
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_3_kernel*'
_output_shapes
:@А*
dtype02
Conv2D/ReadVariableOpЫ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:А*
paddingSAME*
strides
2
Conv2Dc
IdentityIdentityConv2D:output:0*
T0*'
_output_shapes
:А2

Identity"
identityIdentity:output:0*)
_input_shapes
:@::N J
&
_output_shapes
:@
 
_user_specified_nameinputs
х
А
4__inference_batch_normalization_7_layer_call_fn_4818

inputs
batch_normalization_7_gamma
batch_normalization_7_beta%
!batch_normalization_7_moving_mean)
%batch_normalization_7_moving_variance
identityИвStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_7_gammabatch_normalization_7_beta!batch_normalization_7_moving_mean%batch_normalization_7_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:А*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_23232
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:А::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:А
 
_user_specified_nameinputs
х
А
4__inference_batch_normalization_8_layer_call_fn_4949

inputs
batch_normalization_8_gamma
batch_normalization_8_beta%
!batch_normalization_8_moving_mean)
%batch_normalization_8_moving_variance
identityИвStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_8_gammabatch_normalization_8_beta!batch_normalization_8_moving_mean%batch_normalization_8_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:А*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_24122
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:А::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:А
 
_user_specified_nameinputs
т
у
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_4184

inputs.
*readvariableop_batch_normalization_2_gamma/
+readvariableop_1_batch_normalization_2_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_2_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance
identityИЗ
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_2_gamma*
_output_shapes
: *
dtype02
ReadVariableOpМ
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_2_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1└
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp╩
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1┴
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3g
IdentityIdentityFusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": :::::N J
&
_output_shapes
: 
 
_user_specified_nameinputs
х
А
4__inference_batch_normalization_6_layer_call_fn_4754

inputs
batch_normalization_6_gamma
batch_normalization_6_beta%
!batch_normalization_6_moving_mean)
%batch_normalization_6_moving_variance
identityИвStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_6_gammabatch_normalization_6_beta!batch_normalization_6_moving_mean%batch_normalization_6_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:А*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_22432
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:А::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:А
 
_user_specified_nameinputs
М	
и
I__inference_depthwise_conv2d_layer_call_and_return_conditional_losses_828

inputs>
:depthwise_readvariableop_depthwise_conv2d_depthwise_kernel
identityИ╖
depthwise/ReadVariableOpReadVariableOp:depthwise_readvariableop_depthwise_conv2d_depthwise_kernel*&
_output_shapes
: *
dtype02
depthwise/ReadVariableOp{
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2
depthwise/ShapeГ
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
depthwise/dilation_rate╬
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingVALID*
strides
2
	depthwiseА
IdentityIdentitydepthwise:output:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+                            ::i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
с
А
4__inference_batch_normalization_1_layer_call_fn_4071

inputs
batch_normalization_1_gamma
batch_normalization_1_beta%
!batch_normalization_1_moving_mean)
%batch_normalization_1_moving_variance
identityИвStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_1_gammabatch_normalization_1_beta!batch_normalization_1_moving_mean%batch_normalization_1_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_18162
StatefulPartitionedCallН
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": ::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
: 
 
_user_specified_nameinputs"╕L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ъ
serving_defaultЖ
6
input_1+
serving_default_input_1:01(0
dense'
StatefulPartitionedCall:0tensorflow/serving/predict:╦Ё
ШЛ
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
layer-22
layer_with_weights-14
layer-23
layer_with_weights-15
layer-24
layer-25
layer_with_weights-16
layer-26
layer_with_weights-17
layer-27
layer-28
layer-29
layer-30
 layer_with_weights-18
 layer-31
!trainable_variables
"	variables
#regularization_losses
$	keras_api
%
signatures
·_default_save_signature
√__call__
+№&call_and_return_all_conditional_losses"лВ
_tf_keras_networkОВ{"class_name": "Functional", "name": "functional_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 49, 40]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "ExpandDims", "trainable": true, "dtype": "float32", "node_def": {"name": "ExpandDims", "op": "ExpandDims", "input": ["input_1", "ExpandDims/dim"], "attr": {"Tdim": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": 2}}, "name": "tf_op_layer_ExpandDims", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["tf_op_layer_ExpandDims", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 1, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu", "inbound_nodes": [[["batch_normalization", 0, 0, {}]]]}, {"class_name": "DepthwiseConv2D", "config": {"name": "depthwise_conv2d", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "name": "depthwise_conv2d", "inbound_nodes": [[["re_lu", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 1, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1", "inbound_nodes": [[["depthwise_conv2d", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_1", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_1", "inbound_nodes": [[["batch_normalization_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["re_lu_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 1, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_2", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_2", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_2", "inbound_nodes": [[["batch_normalization_2", 0, 0, {}]]]}, {"class_name": "DepthwiseConv2D", "config": {"name": "depthwise_conv2d_1", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "name": "depthwise_conv2d_1", "inbound_nodes": [[["re_lu_2", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 1, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_3", "inbound_nodes": [[["depthwise_conv2d_1", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_3", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_3", "inbound_nodes": [[["batch_normalization_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["re_lu_3", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 1, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_4", "inbound_nodes": [[["conv2d_2", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_4", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_4", "inbound_nodes": [[["batch_normalization_4", 0, 0, {}]]]}, {"class_name": "DepthwiseConv2D", "config": {"name": "depthwise_conv2d_2", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "name": "depthwise_conv2d_2", "inbound_nodes": [[["re_lu_4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 1, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_5", "inbound_nodes": [[["depthwise_conv2d_2", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_5", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_5", "inbound_nodes": [[["batch_normalization_5", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_3", "inbound_nodes": [[["re_lu_5", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 1, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_6", "inbound_nodes": [[["conv2d_3", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_6", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_6", "inbound_nodes": [[["batch_normalization_6", 0, 0, {}]]]}, {"class_name": "DepthwiseConv2D", "config": {"name": "depthwise_conv2d_3", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "name": "depthwise_conv2d_3", "inbound_nodes": [[["re_lu_6", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 1, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_7", "inbound_nodes": [[["depthwise_conv2d_3", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_7", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_7", "inbound_nodes": [[["batch_normalization_7", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_4", "inbound_nodes": [[["re_lu_7", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 1, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_8", "inbound_nodes": [[["conv2d_4", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_8", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_8", "inbound_nodes": [[["batch_normalization_8", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "global_average_pooling2d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_average_pooling2d", "inbound_nodes": [[["re_lu_8", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["global_average_pooling2d", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 14, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["dropout", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 49, 40]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 49, 40]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "ExpandDims", "trainable": true, "dtype": "float32", "node_def": {"name": "ExpandDims", "op": "ExpandDims", "input": ["input_1", "ExpandDims/dim"], "attr": {"Tdim": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": 2}}, "name": "tf_op_layer_ExpandDims", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["tf_op_layer_ExpandDims", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 1, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu", "inbound_nodes": [[["batch_normalization", 0, 0, {}]]]}, {"class_name": "DepthwiseConv2D", "config": {"name": "depthwise_conv2d", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "name": "depthwise_conv2d", "inbound_nodes": [[["re_lu", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 1, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1", "inbound_nodes": [[["depthwise_conv2d", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_1", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_1", "inbound_nodes": [[["batch_normalization_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["re_lu_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 1, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_2", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_2", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_2", "inbound_nodes": [[["batch_normalization_2", 0, 0, {}]]]}, {"class_name": "DepthwiseConv2D", "config": {"name": "depthwise_conv2d_1", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "name": "depthwise_conv2d_1", "inbound_nodes": [[["re_lu_2", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 1, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_3", "inbound_nodes": [[["depthwise_conv2d_1", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_3", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_3", "inbound_nodes": [[["batch_normalization_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["re_lu_3", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 1, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_4", "inbound_nodes": [[["conv2d_2", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_4", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_4", "inbound_nodes": [[["batch_normalization_4", 0, 0, {}]]]}, {"class_name": "DepthwiseConv2D", "config": {"name": "depthwise_conv2d_2", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "name": "depthwise_conv2d_2", "inbound_nodes": [[["re_lu_4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 1, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_5", "inbound_nodes": [[["depthwise_conv2d_2", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_5", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_5", "inbound_nodes": [[["batch_normalization_5", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_3", "inbound_nodes": [[["re_lu_5", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 1, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_6", "inbound_nodes": [[["conv2d_3", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_6", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_6", "inbound_nodes": [[["batch_normalization_6", 0, 0, {}]]]}, {"class_name": "DepthwiseConv2D", "config": {"name": "depthwise_conv2d_3", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "name": "depthwise_conv2d_3", "inbound_nodes": [[["re_lu_6", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 1, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_7", "inbound_nodes": [[["depthwise_conv2d_3", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_7", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_7", "inbound_nodes": [[["batch_normalization_7", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_4", "inbound_nodes": [[["re_lu_7", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 1, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_8", "inbound_nodes": [[["conv2d_4", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_8", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_8", "inbound_nodes": [[["batch_normalization_8", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "global_average_pooling2d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_average_pooling2d", "inbound_nodes": [[["re_lu_8", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["global_average_pooling2d", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 14, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["dropout", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense", 0, 0]]}}}
э"ъ
_tf_keras_input_layer╩{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [1, 49, 40]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 49, 40]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
К
&trainable_variables
'	variables
(regularization_losses
)	keras_api
¤__call__
+■&call_and_return_all_conditional_losses"∙
_tf_keras_layer▀{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_ExpandDims", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "ExpandDims", "trainable": true, "dtype": "float32", "node_def": {"name": "ExpandDims", "op": "ExpandDims", "input": ["input_1", "ExpandDims/dim"], "attr": {"Tdim": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": 2}}}
Щ	

*kernel
+trainable_variables
,	variables
-regularization_losses
.	keras_api
 __call__
+А&call_and_return_all_conditional_losses"№
_tf_keras_layerт{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 40}}}}
ф
/axis
	0gamma
1beta
2moving_mean
3moving_variance
4trainable_variables
5	variables
6regularization_losses
7	keras_api
Б__call__
+В&call_and_return_all_conditional_losses"О
_tf_keras_layerЇ{"class_name": "BatchNormalization", "name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 1, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}}
ш
8trainable_variables
9	variables
:regularization_losses
;	keras_api
Г__call__
+Д&call_and_return_all_conditional_losses"╫
_tf_keras_layer╜{"class_name": "ReLU", "name": "re_lu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}}
╤	
<depthwise_kernel
=trainable_variables
>	variables
?regularization_losses
@	keras_api
Е__call__
+Ж&call_and_return_all_conditional_losses"к
_tf_keras_layerР{"class_name": "DepthwiseConv2D", "name": "depthwise_conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "depthwise_conv2d", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}}
ш
Aaxis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance
Ftrainable_variables
G	variables
Hregularization_losses
I	keras_api
З__call__
+И&call_and_return_all_conditional_losses"Т
_tf_keras_layer°{"class_name": "BatchNormalization", "name": "batch_normalization_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 1, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}}
ь
Jtrainable_variables
K	variables
Lregularization_losses
M	keras_api
Й__call__
+К&call_and_return_all_conditional_losses"█
_tf_keras_layer┴{"class_name": "ReLU", "name": "re_lu_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_1", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}}
Ь	

Nkernel
Otrainable_variables
P	variables
Qregularization_losses
R	keras_api
Л__call__
+М&call_and_return_all_conditional_losses" 
_tf_keras_layerх{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}}
ш
Saxis
	Tgamma
Ubeta
Vmoving_mean
Wmoving_variance
Xtrainable_variables
Y	variables
Zregularization_losses
[	keras_api
Н__call__
+О&call_and_return_all_conditional_losses"Т
_tf_keras_layer°{"class_name": "BatchNormalization", "name": "batch_normalization_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 1, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}}
ь
\trainable_variables
]	variables
^regularization_losses
_	keras_api
П__call__
+Р&call_and_return_all_conditional_losses"█
_tf_keras_layer┴{"class_name": "ReLU", "name": "re_lu_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_2", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}}
╒	
`depthwise_kernel
atrainable_variables
b	variables
cregularization_losses
d	keras_api
С__call__
+Т&call_and_return_all_conditional_losses"о
_tf_keras_layerФ{"class_name": "DepthwiseConv2D", "name": "depthwise_conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "depthwise_conv2d_1", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}}
ш
eaxis
	fgamma
gbeta
hmoving_mean
imoving_variance
jtrainable_variables
k	variables
lregularization_losses
m	keras_api
У__call__
+Ф&call_and_return_all_conditional_losses"Т
_tf_keras_layer°{"class_name": "BatchNormalization", "name": "batch_normalization_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 1, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}}
ь
ntrainable_variables
o	variables
pregularization_losses
q	keras_api
Х__call__
+Ц&call_and_return_all_conditional_losses"█
_tf_keras_layer┴{"class_name": "ReLU", "name": "re_lu_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_3", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}}
Ь	

rkernel
strainable_variables
t	variables
uregularization_losses
v	keras_api
Ч__call__
+Ш&call_and_return_all_conditional_losses" 
_tf_keras_layerх{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}}
ш
waxis
	xgamma
ybeta
zmoving_mean
{moving_variance
|trainable_variables
}	variables
~regularization_losses
	keras_api
Щ__call__
+Ъ&call_and_return_all_conditional_losses"Т
_tf_keras_layer°{"class_name": "BatchNormalization", "name": "batch_normalization_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 1, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}}
Ё
Аtrainable_variables
Б	variables
Вregularization_losses
Г	keras_api
Ы__call__
+Ь&call_and_return_all_conditional_losses"█
_tf_keras_layer┴{"class_name": "ReLU", "name": "re_lu_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_4", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}}
┘	
Дdepthwise_kernel
Еtrainable_variables
Ж	variables
Зregularization_losses
И	keras_api
Э__call__
+Ю&call_and_return_all_conditional_losses"н
_tf_keras_layerУ{"class_name": "DepthwiseConv2D", "name": "depthwise_conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "depthwise_conv2d_2", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
ё
	Йaxis

Кgamma
	Лbeta
Мmoving_mean
Нmoving_variance
Оtrainable_variables
П	variables
Рregularization_losses
С	keras_api
Я__call__
+а&call_and_return_all_conditional_losses"Т
_tf_keras_layer°{"class_name": "BatchNormalization", "name": "batch_normalization_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 1, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}}
Ё
Тtrainable_variables
У	variables
Фregularization_losses
Х	keras_api
б__call__
+в&call_and_return_all_conditional_losses"█
_tf_keras_layer┴{"class_name": "ReLU", "name": "re_lu_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_5", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}}
в	
Цkernel
Чtrainable_variables
Ш	variables
Щregularization_losses
Ъ	keras_api
г__call__
+д&call_and_return_all_conditional_losses"А
_tf_keras_layerц{"class_name": "Conv2D", "name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}}
Є
	Ыaxis

Ьgamma
	Эbeta
Юmoving_mean
Яmoving_variance
аtrainable_variables
б	variables
вregularization_losses
г	keras_api
е__call__
+ж&call_and_return_all_conditional_losses"У
_tf_keras_layer∙{"class_name": "BatchNormalization", "name": "batch_normalization_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 1, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}}
Ё
дtrainable_variables
е	variables
жregularization_losses
з	keras_api
з__call__
+и&call_and_return_all_conditional_losses"█
_tf_keras_layer┴{"class_name": "ReLU", "name": "re_lu_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_6", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}}
┌	
иdepthwise_kernel
йtrainable_variables
к	variables
лregularization_losses
м	keras_api
й__call__
+к&call_and_return_all_conditional_losses"о
_tf_keras_layerФ{"class_name": "DepthwiseConv2D", "name": "depthwise_conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "depthwise_conv2d_3", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}}
Є
	нaxis

оgamma
	пbeta
░moving_mean
▒moving_variance
▓trainable_variables
│	variables
┤regularization_losses
╡	keras_api
л__call__
+м&call_and_return_all_conditional_losses"У
_tf_keras_layer∙{"class_name": "BatchNormalization", "name": "batch_normalization_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 1, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}}
Ё
╢trainable_variables
╖	variables
╕regularization_losses
╣	keras_api
н__call__
+о&call_and_return_all_conditional_losses"█
_tf_keras_layer┴{"class_name": "ReLU", "name": "re_lu_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_7", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}}
г	
║kernel
╗trainable_variables
╝	variables
╜regularization_losses
╛	keras_api
п__call__
+░&call_and_return_all_conditional_losses"Б
_tf_keras_layerч{"class_name": "Conv2D", "name": "conv2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}}
Є
	┐axis

└gamma
	┴beta
┬moving_mean
├moving_variance
─trainable_variables
┼	variables
╞regularization_losses
╟	keras_api
▒__call__
+▓&call_and_return_all_conditional_losses"У
_tf_keras_layer∙{"class_name": "BatchNormalization", "name": "batch_normalization_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 1, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}}
Ё
╚trainable_variables
╔	variables
╩regularization_losses
╦	keras_api
│__call__
+┤&call_and_return_all_conditional_losses"█
_tf_keras_layer┴{"class_name": "ReLU", "name": "re_lu_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_8", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}}
Щ
╠trainable_variables
═	variables
╬regularization_losses
╧	keras_api
╡__call__
+╢&call_and_return_all_conditional_losses"Д
_tf_keras_layerъ{"class_name": "GlobalAveragePooling2D", "name": "global_average_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "global_average_pooling2d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ч
╨trainable_variables
╤	variables
╥regularization_losses
╙	keras_api
╖__call__
+╕&call_and_return_all_conditional_losses"╥
_tf_keras_layer╕{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
о
╘kernel
	╒bias
╓trainable_variables
╫	variables
╪regularization_losses
┘	keras_api
╣__call__
+║&call_and_return_all_conditional_losses"Б
_tf_keras_layerч{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 14, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}
М
*0
01
12
<3
B4
C5
N6
T7
U8
`9
f10
g11
r12
x13
y14
Д15
К16
Л17
Ц18
Ь19
Э20
и21
о22
п23
║24
└25
┴26
╘27
╒28"
trackable_list_wrapper
д
*0
01
12
23
34
<5
B6
C7
D8
E9
N10
T11
U12
V13
W14
`15
f16
g17
h18
i19
r20
x21
y22
z23
{24
Д25
К26
Л27
М28
Н29
Ц30
Ь31
Э32
Ю33
Я34
и35
о36
п37
░38
▒39
║40
└41
┴42
┬43
├44
╘45
╒46"
trackable_list_wrapper
 "
trackable_list_wrapper
╙
┌metrics
█layers
!trainable_variables
▄layer_metrics
▌non_trainable_variables
 ▐layer_regularization_losses
"	variables
#regularization_losses
√__call__
·_default_save_signature
+№&call_and_return_all_conditional_losses
'№"call_and_return_conditional_losses"
_generic_user_object
-
╗serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
▀metrics
рlayers
&trainable_variables
сlayer_metrics
тnon_trainable_variables
 уlayer_regularization_losses
'	variables
(regularization_losses
¤__call__
+■&call_and_return_all_conditional_losses
'■"call_and_return_conditional_losses"
_generic_user_object
':%( 2conv2d/kernel
'
*0"
trackable_list_wrapper
'
*0"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
фmetrics
хlayers
+trainable_variables
цlayer_metrics
чnon_trainable_variables
 шlayer_regularization_losses
,	variables
-regularization_losses
 __call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':% 2batch_normalization/gamma
&:$ 2batch_normalization/beta
/:-  (2batch_normalization/moving_mean
3:1  (2#batch_normalization/moving_variance
.
00
11"
trackable_list_wrapper
<
00
11
22
33"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
щmetrics
ъlayers
4trainable_variables
ыlayer_metrics
ьnon_trainable_variables
 эlayer_regularization_losses
5	variables
6regularization_losses
Б__call__
+В&call_and_return_all_conditional_losses
'В"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
юmetrics
яlayers
8trainable_variables
Ёlayer_metrics
ёnon_trainable_variables
 Єlayer_regularization_losses
9	variables
:regularization_losses
Г__call__
+Д&call_and_return_all_conditional_losses
'Д"call_and_return_conditional_losses"
_generic_user_object
;:9 2!depthwise_conv2d/depthwise_kernel
'
<0"
trackable_list_wrapper
'
<0"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
єmetrics
Їlayers
=trainable_variables
їlayer_metrics
Ўnon_trainable_variables
 ўlayer_regularization_losses
>	variables
?regularization_losses
Е__call__
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):' 2batch_normalization_1/gamma
(:& 2batch_normalization_1/beta
1:/  (2!batch_normalization_1/moving_mean
5:3  (2%batch_normalization_1/moving_variance
.
B0
C1"
trackable_list_wrapper
<
B0
C1
D2
E3"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
°metrics
∙layers
Ftrainable_variables
·layer_metrics
√non_trainable_variables
 №layer_regularization_losses
G	variables
Hregularization_losses
З__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
¤metrics
■layers
Jtrainable_variables
 layer_metrics
Аnon_trainable_variables
 Бlayer_regularization_losses
K	variables
Lregularization_losses
Й__call__
+К&call_and_return_all_conditional_losses
'К"call_and_return_conditional_losses"
_generic_user_object
):'  2conv2d_1/kernel
'
N0"
trackable_list_wrapper
'
N0"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
Вmetrics
Гlayers
Otrainable_variables
Дlayer_metrics
Еnon_trainable_variables
 Жlayer_regularization_losses
P	variables
Qregularization_losses
Л__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):' 2batch_normalization_2/gamma
(:& 2batch_normalization_2/beta
1:/  (2!batch_normalization_2/moving_mean
5:3  (2%batch_normalization_2/moving_variance
.
T0
U1"
trackable_list_wrapper
<
T0
U1
V2
W3"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
Зmetrics
Иlayers
Xtrainable_variables
Йlayer_metrics
Кnon_trainable_variables
 Лlayer_regularization_losses
Y	variables
Zregularization_losses
Н__call__
+О&call_and_return_all_conditional_losses
'О"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
Мmetrics
Нlayers
\trainable_variables
Оlayer_metrics
Пnon_trainable_variables
 Рlayer_regularization_losses
]	variables
^regularization_losses
П__call__
+Р&call_and_return_all_conditional_losses
'Р"call_and_return_conditional_losses"
_generic_user_object
=:; 2#depthwise_conv2d_1/depthwise_kernel
'
`0"
trackable_list_wrapper
'
`0"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
Сmetrics
Тlayers
atrainable_variables
Уlayer_metrics
Фnon_trainable_variables
 Хlayer_regularization_losses
b	variables
cregularization_losses
С__call__
+Т&call_and_return_all_conditional_losses
'Т"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):' 2batch_normalization_3/gamma
(:& 2batch_normalization_3/beta
1:/  (2!batch_normalization_3/moving_mean
5:3  (2%batch_normalization_3/moving_variance
.
f0
g1"
trackable_list_wrapper
<
f0
g1
h2
i3"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
Цmetrics
Чlayers
jtrainable_variables
Шlayer_metrics
Щnon_trainable_variables
 Ъlayer_regularization_losses
k	variables
lregularization_losses
У__call__
+Ф&call_and_return_all_conditional_losses
'Ф"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
Ыmetrics
Ьlayers
ntrainable_variables
Эlayer_metrics
Юnon_trainable_variables
 Яlayer_regularization_losses
o	variables
pregularization_losses
Х__call__
+Ц&call_and_return_all_conditional_losses
'Ц"call_and_return_conditional_losses"
_generic_user_object
):' @2conv2d_2/kernel
'
r0"
trackable_list_wrapper
'
r0"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
аmetrics
бlayers
strainable_variables
вlayer_metrics
гnon_trainable_variables
 дlayer_regularization_losses
t	variables
uregularization_losses
Ч__call__
+Ш&call_and_return_all_conditional_losses
'Ш"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2batch_normalization_4/gamma
(:&@2batch_normalization_4/beta
1:/@ (2!batch_normalization_4/moving_mean
5:3@ (2%batch_normalization_4/moving_variance
.
x0
y1"
trackable_list_wrapper
<
x0
y1
z2
{3"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
еmetrics
жlayers
|trainable_variables
зlayer_metrics
иnon_trainable_variables
 йlayer_regularization_losses
}	variables
~regularization_losses
Щ__call__
+Ъ&call_and_return_all_conditional_losses
'Ъ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
кmetrics
лlayers
Аtrainable_variables
мlayer_metrics
нnon_trainable_variables
 оlayer_regularization_losses
Б	variables
Вregularization_losses
Ы__call__
+Ь&call_and_return_all_conditional_losses
'Ь"call_and_return_conditional_losses"
_generic_user_object
=:;@2#depthwise_conv2d_2/depthwise_kernel
(
Д0"
trackable_list_wrapper
(
Д0"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
пmetrics
░layers
Еtrainable_variables
▒layer_metrics
▓non_trainable_variables
 │layer_regularization_losses
Ж	variables
Зregularization_losses
Э__call__
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2batch_normalization_5/gamma
(:&@2batch_normalization_5/beta
1:/@ (2!batch_normalization_5/moving_mean
5:3@ (2%batch_normalization_5/moving_variance
0
К0
Л1"
trackable_list_wrapper
@
К0
Л1
М2
Н3"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
┤metrics
╡layers
Оtrainable_variables
╢layer_metrics
╖non_trainable_variables
 ╕layer_regularization_losses
П	variables
Рregularization_losses
Я__call__
+а&call_and_return_all_conditional_losses
'а"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╣metrics
║layers
Тtrainable_variables
╗layer_metrics
╝non_trainable_variables
 ╜layer_regularization_losses
У	variables
Фregularization_losses
б__call__
+в&call_and_return_all_conditional_losses
'в"call_and_return_conditional_losses"
_generic_user_object
*:(@А2conv2d_3/kernel
(
Ц0"
trackable_list_wrapper
(
Ц0"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╛metrics
┐layers
Чtrainable_variables
└layer_metrics
┴non_trainable_variables
 ┬layer_regularization_losses
Ш	variables
Щregularization_losses
г__call__
+д&call_and_return_all_conditional_losses
'д"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(А2batch_normalization_6/gamma
):'А2batch_normalization_6/beta
2:0А (2!batch_normalization_6/moving_mean
6:4А (2%batch_normalization_6/moving_variance
0
Ь0
Э1"
trackable_list_wrapper
@
Ь0
Э1
Ю2
Я3"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
├metrics
─layers
аtrainable_variables
┼layer_metrics
╞non_trainable_variables
 ╟layer_regularization_losses
б	variables
вregularization_losses
е__call__
+ж&call_and_return_all_conditional_losses
'ж"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╚metrics
╔layers
дtrainable_variables
╩layer_metrics
╦non_trainable_variables
 ╠layer_regularization_losses
е	variables
жregularization_losses
з__call__
+и&call_and_return_all_conditional_losses
'и"call_and_return_conditional_losses"
_generic_user_object
>:<А2#depthwise_conv2d_3/depthwise_kernel
(
и0"
trackable_list_wrapper
(
и0"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
═metrics
╬layers
йtrainable_variables
╧layer_metrics
╨non_trainable_variables
 ╤layer_regularization_losses
к	variables
лregularization_losses
й__call__
+к&call_and_return_all_conditional_losses
'к"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(А2batch_normalization_7/gamma
):'А2batch_normalization_7/beta
2:0А (2!batch_normalization_7/moving_mean
6:4А (2%batch_normalization_7/moving_variance
0
о0
п1"
trackable_list_wrapper
@
о0
п1
░2
▒3"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╥metrics
╙layers
▓trainable_variables
╘layer_metrics
╒non_trainable_variables
 ╓layer_regularization_losses
│	variables
┤regularization_losses
л__call__
+м&call_and_return_all_conditional_losses
'м"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╫metrics
╪layers
╢trainable_variables
┘layer_metrics
┌non_trainable_variables
 █layer_regularization_losses
╖	variables
╕regularization_losses
н__call__
+о&call_and_return_all_conditional_losses
'о"call_and_return_conditional_losses"
_generic_user_object
+:)АА2conv2d_4/kernel
(
║0"
trackable_list_wrapper
(
║0"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
▄metrics
▌layers
╗trainable_variables
▐layer_metrics
▀non_trainable_variables
 рlayer_regularization_losses
╝	variables
╜regularization_losses
п__call__
+░&call_and_return_all_conditional_losses
'░"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(А2batch_normalization_8/gamma
):'А2batch_normalization_8/beta
2:0А (2!batch_normalization_8/moving_mean
6:4А (2%batch_normalization_8/moving_variance
0
└0
┴1"
trackable_list_wrapper
@
└0
┴1
┬2
├3"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
сmetrics
тlayers
─trainable_variables
уlayer_metrics
фnon_trainable_variables
 хlayer_regularization_losses
┼	variables
╞regularization_losses
▒__call__
+▓&call_and_return_all_conditional_losses
'▓"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
цmetrics
чlayers
╚trainable_variables
шlayer_metrics
щnon_trainable_variables
 ъlayer_regularization_losses
╔	variables
╩regularization_losses
│__call__
+┤&call_and_return_all_conditional_losses
'┤"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
ыmetrics
ьlayers
╠trainable_variables
эlayer_metrics
юnon_trainable_variables
 яlayer_regularization_losses
═	variables
╬regularization_losses
╡__call__
+╢&call_and_return_all_conditional_losses
'╢"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Ёmetrics
ёlayers
╨trainable_variables
Єlayer_metrics
єnon_trainable_variables
 Їlayer_regularization_losses
╤	variables
╥regularization_losses
╖__call__
+╕&call_and_return_all_conditional_losses
'╕"call_and_return_conditional_losses"
_generic_user_object
:	А2dense/kernel
:2
dense/bias
0
╘0
╒1"
trackable_list_wrapper
0
╘0
╒1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
їmetrics
Ўlayers
╓trainable_variables
ўlayer_metrics
°non_trainable_variables
 ∙layer_regularization_losses
╫	variables
╪regularization_losses
╣__call__
+║&call_and_return_all_conditional_losses
'║"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
Ц
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
 31"
trackable_list_wrapper
 "
trackable_dict_wrapper
о
20
31
D2
E3
V4
W5
h6
i7
z8
{9
М10
Н11
Ю12
Я13
░14
▒15
┬16
├17"
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
.
20
31"
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
.
D0
E1"
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
.
V0
W1"
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
.
h0
i1"
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
.
z0
{1"
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
0
М0
Н1"
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
0
Ю0
Я1"
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
0
░0
▒1"
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
0
┬0
├1"
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
р2▌
__inference__wrapped_model_714║
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк **в'
%К"
input_1         1(
·2ў
+__inference_functional_1_layer_call_fn_3875
+__inference_functional_1_layer_call_fn_3342
+__inference_functional_1_layer_call_fn_3394
+__inference_functional_1_layer_call_fn_3823└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ц2у
F__inference_functional_1_layer_call_and_return_conditional_losses_3290
F__inference_functional_1_layer_call_and_return_conditional_losses_3771
F__inference_functional_1_layer_call_and_return_conditional_losses_3105
F__inference_functional_1_layer_call_and_return_conditional_losses_3586└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
▀2▄
5__inference_tf_op_layer_ExpandDims_layer_call_fn_3886в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
·2ў
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_3881в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╧2╠
%__inference_conv2d_layer_call_fn_3899в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ъ2ч
@__inference_conv2d_layer_call_and_return_conditional_losses_3893в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
К2З
2__inference_batch_normalization_layer_call_fn_3998
2__inference_batch_normalization_layer_call_fn_4007
2__inference_batch_normalization_layer_call_fn_3953
2__inference_batch_normalization_layer_call_fn_3944┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ў2є
M__inference_batch_normalization_layer_call_and_return_conditional_losses_3917
M__inference_batch_normalization_layer_call_and_return_conditional_losses_3935
M__inference_batch_normalization_layer_call_and_return_conditional_losses_3989
M__inference_batch_normalization_layer_call_and_return_conditional_losses_3971┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╬2╦
$__inference_re_lu_layer_call_fn_4017в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
щ2ц
?__inference_re_lu_layer_call_and_return_conditional_losses_4012в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Н2К
.__inference_depthwise_conv2d_layer_call_fn_832╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                            
и2е
I__inference_depthwise_conv2d_layer_call_and_return_conditional_losses_816╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                            
Т2П
4__inference_batch_normalization_1_layer_call_fn_4071
4__inference_batch_normalization_1_layer_call_fn_4125
4__inference_batch_normalization_1_layer_call_fn_4116
4__inference_batch_normalization_1_layer_call_fn_4062┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
■2√
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4089
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4053
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4107
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4035┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╨2═
&__inference_re_lu_1_layer_call_fn_4135в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ы2ш
A__inference_re_lu_1_layer_call_and_return_conditional_losses_4130в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╤2╬
'__inference_conv2d_1_layer_call_fn_4148в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ь2щ
B__inference_conv2d_1_layer_call_and_return_conditional_losses_4142в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Т2П
4__inference_batch_normalization_2_layer_call_fn_4193
4__inference_batch_normalization_2_layer_call_fn_4247
4__inference_batch_normalization_2_layer_call_fn_4256
4__inference_batch_normalization_2_layer_call_fn_4202┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
■2√
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_4220
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_4184
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_4238
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_4166┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╨2═
&__inference_re_lu_2_layer_call_fn_4266в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ы2ш
A__inference_re_lu_2_layer_call_and_return_conditional_losses_4261в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Р2Н
1__inference_depthwise_conv2d_1_layer_call_fn_1042╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                            
л2и
L__inference_depthwise_conv2d_1_layer_call_and_return_conditional_losses_1026╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                            
Т2П
4__inference_batch_normalization_3_layer_call_fn_4320
4__inference_batch_normalization_3_layer_call_fn_4311
4__inference_batch_normalization_3_layer_call_fn_4365
4__inference_batch_normalization_3_layer_call_fn_4374┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
■2√
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_4302
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_4338
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_4284
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_4356┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╨2═
&__inference_re_lu_3_layer_call_fn_4384в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ы2ш
A__inference_re_lu_3_layer_call_and_return_conditional_losses_4379в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╤2╬
'__inference_conv2d_2_layer_call_fn_4397в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ь2щ
B__inference_conv2d_2_layer_call_and_return_conditional_losses_4391в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Т2П
4__inference_batch_normalization_4_layer_call_fn_4505
4__inference_batch_normalization_4_layer_call_fn_4442
4__inference_batch_normalization_4_layer_call_fn_4451
4__inference_batch_normalization_4_layer_call_fn_4496┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
■2√
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_4433
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_4415
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_4469
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_4487┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╨2═
&__inference_re_lu_4_layer_call_fn_4515в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ы2ш
A__inference_re_lu_4_layer_call_and_return_conditional_losses_4510в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Р2Н
1__inference_depthwise_conv2d_2_layer_call_fn_1252╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                           @
л2и
L__inference_depthwise_conv2d_2_layer_call_and_return_conditional_losses_1236╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                           @
Т2П
4__inference_batch_normalization_5_layer_call_fn_4560
4__inference_batch_normalization_5_layer_call_fn_4623
4__inference_batch_normalization_5_layer_call_fn_4569
4__inference_batch_normalization_5_layer_call_fn_4614┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
■2√
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_4605
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_4587
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_4533
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_4551┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╨2═
&__inference_re_lu_5_layer_call_fn_4633в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ы2ш
A__inference_re_lu_5_layer_call_and_return_conditional_losses_4628в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╤2╬
'__inference_conv2d_3_layer_call_fn_4646в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ь2щ
B__inference_conv2d_3_layer_call_and_return_conditional_losses_4640в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Т2П
4__inference_batch_normalization_6_layer_call_fn_4754
4__inference_batch_normalization_6_layer_call_fn_4691
4__inference_batch_normalization_6_layer_call_fn_4745
4__inference_batch_normalization_6_layer_call_fn_4700┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
■2√
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_4736
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_4682
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_4664
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_4718┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╨2═
&__inference_re_lu_6_layer_call_fn_4764в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ы2ш
A__inference_re_lu_6_layer_call_and_return_conditional_losses_4759в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
С2О
1__inference_depthwise_conv2d_3_layer_call_fn_1462╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           А
м2й
L__inference_depthwise_conv2d_3_layer_call_and_return_conditional_losses_1446╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           А
Т2П
4__inference_batch_normalization_7_layer_call_fn_4872
4__inference_batch_normalization_7_layer_call_fn_4818
4__inference_batch_normalization_7_layer_call_fn_4863
4__inference_batch_normalization_7_layer_call_fn_4809┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
■2√
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_4800
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_4836
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_4854
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_4782┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╨2═
&__inference_re_lu_7_layer_call_fn_4882в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ы2ш
A__inference_re_lu_7_layer_call_and_return_conditional_losses_4877в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╤2╬
'__inference_conv2d_4_layer_call_fn_4895в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ь2щ
B__inference_conv2d_4_layer_call_and_return_conditional_losses_4889в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Т2П
4__inference_batch_normalization_8_layer_call_fn_5003
4__inference_batch_normalization_8_layer_call_fn_4949
4__inference_batch_normalization_8_layer_call_fn_4940
4__inference_batch_normalization_8_layer_call_fn_4994┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
■2√
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_4967
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_4985
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_4913
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_4931┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╨2═
&__inference_re_lu_8_layer_call_fn_5013в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ы2ш
A__inference_re_lu_8_layer_call_and_return_conditional_losses_5008в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Я2Ь
7__inference_global_average_pooling2d_layer_call_fn_1665р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
║2╖
R__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_1653р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
К2З
&__inference_dropout_layer_call_fn_5040
&__inference_dropout_layer_call_fn_5035┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
└2╜
A__inference_dropout_layer_call_and_return_conditional_losses_5030
A__inference_dropout_layer_call_and_return_conditional_losses_5025┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╬2╦
$__inference_dense_layer_call_fn_5057в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
щ2ц
?__inference_dense_layer_call_and_return_conditional_losses_5050в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
1B/
"__inference_signature_wrapper_2913input_1╞
__inference__wrapped_model_714гE*0123<BCDENTUVW`fghirxyz{ДКЛМНЦЬЭЮЯиоп░▒║└┴┬├╘╒4в1
*в'
%К"
input_1         1(
к "$к!

denseК
dense│
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4035`BCDE2в/
(в%
К
inputs 
p
к "$в!
К
0 
Ъ │
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4053`BCDE2в/
(в%
К
inputs 
p 
к "$в!
К
0 
Ъ ъ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4089ЦBCDEMвJ
Cв@
:К7
inputs+                            
p
к "?в<
5К2
0+                            
Ъ ъ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4107ЦBCDEMвJ
Cв@
:К7
inputs+                            
p 
к "?в<
5К2
0+                            
Ъ Л
4__inference_batch_normalization_1_layer_call_fn_4062SBCDE2в/
(в%
К
inputs 
p
к "К Л
4__inference_batch_normalization_1_layer_call_fn_4071SBCDE2в/
(в%
К
inputs 
p 
к "К ┬
4__inference_batch_normalization_1_layer_call_fn_4116ЙBCDEMвJ
Cв@
:К7
inputs+                            
p
к "2К/+                            ┬
4__inference_batch_normalization_1_layer_call_fn_4125ЙBCDEMвJ
Cв@
:К7
inputs+                            
p 
к "2К/+                            │
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_4166`TUVW2в/
(в%
К
inputs 
p
к "$в!
К
0 
Ъ │
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_4184`TUVW2в/
(в%
К
inputs 
p 
к "$в!
К
0 
Ъ ъ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_4220ЦTUVWMвJ
Cв@
:К7
inputs+                            
p
к "?в<
5К2
0+                            
Ъ ъ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_4238ЦTUVWMвJ
Cв@
:К7
inputs+                            
p 
к "?в<
5К2
0+                            
Ъ Л
4__inference_batch_normalization_2_layer_call_fn_4193STUVW2в/
(в%
К
inputs 
p
к "К Л
4__inference_batch_normalization_2_layer_call_fn_4202STUVW2в/
(в%
К
inputs 
p 
к "К ┬
4__inference_batch_normalization_2_layer_call_fn_4247ЙTUVWMвJ
Cв@
:К7
inputs+                            
p
к "2К/+                            ┬
4__inference_batch_normalization_2_layer_call_fn_4256ЙTUVWMвJ
Cв@
:К7
inputs+                            
p 
к "2К/+                            ъ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_4284ЦfghiMвJ
Cв@
:К7
inputs+                            
p
к "?в<
5К2
0+                            
Ъ ъ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_4302ЦfghiMвJ
Cв@
:К7
inputs+                            
p 
к "?в<
5К2
0+                            
Ъ │
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_4338`fghi2в/
(в%
К
inputs 
p
к "$в!
К
0 
Ъ │
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_4356`fghi2в/
(в%
К
inputs 
p 
к "$в!
К
0 
Ъ ┬
4__inference_batch_normalization_3_layer_call_fn_4311ЙfghiMвJ
Cв@
:К7
inputs+                            
p
к "2К/+                            ┬
4__inference_batch_normalization_3_layer_call_fn_4320ЙfghiMвJ
Cв@
:К7
inputs+                            
p 
к "2К/+                            Л
4__inference_batch_normalization_3_layer_call_fn_4365Sfghi2в/
(в%
К
inputs 
p
к "К Л
4__inference_batch_normalization_3_layer_call_fn_4374Sfghi2в/
(в%
К
inputs 
p 
к "К │
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_4415`xyz{2в/
(в%
К
inputs@
p
к "$в!
К
0@
Ъ │
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_4433`xyz{2в/
(в%
К
inputs@
p 
к "$в!
К
0@
Ъ ъ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_4469Цxyz{MвJ
Cв@
:К7
inputs+                           @
p
к "?в<
5К2
0+                           @
Ъ ъ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_4487Цxyz{MвJ
Cв@
:К7
inputs+                           @
p 
к "?в<
5К2
0+                           @
Ъ Л
4__inference_batch_normalization_4_layer_call_fn_4442Sxyz{2в/
(в%
К
inputs@
p
к "К@Л
4__inference_batch_normalization_4_layer_call_fn_4451Sxyz{2в/
(в%
К
inputs@
p 
к "К@┬
4__inference_batch_normalization_4_layer_call_fn_4496Йxyz{MвJ
Cв@
:К7
inputs+                           @
p
к "2К/+                           @┬
4__inference_batch_normalization_4_layer_call_fn_4505Йxyz{MвJ
Cв@
:К7
inputs+                           @
p 
к "2К/+                           @╖
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_4533dКЛМН2в/
(в%
К
inputs@
p
к "$в!
К
0@
Ъ ╖
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_4551dКЛМН2в/
(в%
К
inputs@
p 
к "$в!
К
0@
Ъ ю
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_4587ЪКЛМНMвJ
Cв@
:К7
inputs+                           @
p
к "?в<
5К2
0+                           @
Ъ ю
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_4605ЪКЛМНMвJ
Cв@
:К7
inputs+                           @
p 
к "?в<
5К2
0+                           @
Ъ П
4__inference_batch_normalization_5_layer_call_fn_4560WКЛМН2в/
(в%
К
inputs@
p
к "К@П
4__inference_batch_normalization_5_layer_call_fn_4569WКЛМН2в/
(в%
К
inputs@
p 
к "К@╞
4__inference_batch_normalization_5_layer_call_fn_4614НКЛМНMвJ
Cв@
:К7
inputs+                           @
p
к "2К/+                           @╞
4__inference_batch_normalization_5_layer_call_fn_4623НКЛМНMвJ
Cв@
:К7
inputs+                           @
p 
к "2К/+                           @Ё
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_4664ЬЬЭЮЯNвK
DвA
;К8
inputs,                           А
p
к "@в=
6К3
0,                           А
Ъ Ё
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_4682ЬЬЭЮЯNвK
DвA
;К8
inputs,                           А
p 
к "@в=
6К3
0,                           А
Ъ ╣
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_4718fЬЭЮЯ3в0
)в&
 К
inputsА
p
к "%в"
К
0А
Ъ ╣
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_4736fЬЭЮЯ3в0
)в&
 К
inputsА
p 
к "%в"
К
0А
Ъ ╚
4__inference_batch_normalization_6_layer_call_fn_4691ПЬЭЮЯNвK
DвA
;К8
inputs,                           А
p
к "3К0,                           А╚
4__inference_batch_normalization_6_layer_call_fn_4700ПЬЭЮЯNвK
DвA
;К8
inputs,                           А
p 
к "3К0,                           АС
4__inference_batch_normalization_6_layer_call_fn_4745YЬЭЮЯ3в0
)в&
 К
inputsА
p
к "КАС
4__inference_batch_normalization_6_layer_call_fn_4754YЬЭЮЯ3в0
)в&
 К
inputsА
p 
к "КА╣
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_4782fоп░▒3в0
)в&
 К
inputsА
p
к "%в"
К
0А
Ъ ╣
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_4800fоп░▒3в0
)в&
 К
inputsА
p 
к "%в"
К
0А
Ъ Ё
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_4836Ьоп░▒NвK
DвA
;К8
inputs,                           А
p
к "@в=
6К3
0,                           А
Ъ Ё
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_4854Ьоп░▒NвK
DвA
;К8
inputs,                           А
p 
к "@в=
6К3
0,                           А
Ъ С
4__inference_batch_normalization_7_layer_call_fn_4809Yоп░▒3в0
)в&
 К
inputsА
p
к "КАС
4__inference_batch_normalization_7_layer_call_fn_4818Yоп░▒3в0
)в&
 К
inputsА
p 
к "КА╚
4__inference_batch_normalization_7_layer_call_fn_4863Поп░▒NвK
DвA
;К8
inputs,                           А
p
к "3К0,                           А╚
4__inference_batch_normalization_7_layer_call_fn_4872Поп░▒NвK
DвA
;К8
inputs,                           А
p 
к "3К0,                           А╣
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_4913f└┴┬├3в0
)в&
 К
inputsА
p
к "%в"
К
0А
Ъ ╣
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_4931f└┴┬├3в0
)в&
 К
inputsА
p 
к "%в"
К
0А
Ъ Ё
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_4967Ь└┴┬├NвK
DвA
;К8
inputs,                           А
p
к "@в=
6К3
0,                           А
Ъ Ё
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_4985Ь└┴┬├NвK
DвA
;К8
inputs,                           А
p 
к "@в=
6К3
0,                           А
Ъ С
4__inference_batch_normalization_8_layer_call_fn_4940Y└┴┬├3в0
)в&
 К
inputsА
p
к "КАС
4__inference_batch_normalization_8_layer_call_fn_4949Y└┴┬├3в0
)в&
 К
inputsА
p 
к "КА╚
4__inference_batch_normalization_8_layer_call_fn_4994П└┴┬├NвK
DвA
;К8
inputs,                           А
p
к "3К0,                           А╚
4__inference_batch_normalization_8_layer_call_fn_5003П└┴┬├NвK
DвA
;К8
inputs,                           А
p 
к "3К0,                           Аш
M__inference_batch_normalization_layer_call_and_return_conditional_losses_3917Ц0123MвJ
Cв@
:К7
inputs+                            
p
к "?в<
5К2
0+                            
Ъ ш
M__inference_batch_normalization_layer_call_and_return_conditional_losses_3935Ц0123MвJ
Cв@
:К7
inputs+                            
p 
к "?в<
5К2
0+                            
Ъ ▒
M__inference_batch_normalization_layer_call_and_return_conditional_losses_3971`01232в/
(в%
К
inputs 
p
к "$в!
К
0 
Ъ ▒
M__inference_batch_normalization_layer_call_and_return_conditional_losses_3989`01232в/
(в%
К
inputs 
p 
к "$в!
К
0 
Ъ └
2__inference_batch_normalization_layer_call_fn_3944Й0123MвJ
Cв@
:К7
inputs+                            
p
к "2К/+                            └
2__inference_batch_normalization_layer_call_fn_3953Й0123MвJ
Cв@
:К7
inputs+                            
p 
к "2К/+                            Й
2__inference_batch_normalization_layer_call_fn_3998S01232в/
(в%
К
inputs 
p
к "К Й
2__inference_batch_normalization_layer_call_fn_4007S01232в/
(в%
К
inputs 
p 
к "К Я
B__inference_conv2d_1_layer_call_and_return_conditional_losses_4142YN.в+
$в!
К
inputs 
к "$в!
К
0 
Ъ w
'__inference_conv2d_1_layer_call_fn_4148LN.в+
$в!
К
inputs 
к "К Я
B__inference_conv2d_2_layer_call_and_return_conditional_losses_4391Yr.в+
$в!
К
inputs 
к "$в!
К
0@
Ъ w
'__inference_conv2d_2_layer_call_fn_4397Lr.в+
$в!
К
inputs 
к "К@б
B__inference_conv2d_3_layer_call_and_return_conditional_losses_4640[Ц.в+
$в!
К
inputs@
к "%в"
К
0А
Ъ y
'__inference_conv2d_3_layer_call_fn_4646NЦ.в+
$в!
К
inputs@
к "КАв
B__inference_conv2d_4_layer_call_and_return_conditional_losses_4889\║/в,
%в"
 К
inputsА
к "%в"
К
0А
Ъ z
'__inference_conv2d_4_layer_call_fn_4895O║/в,
%в"
 К
inputsА
к "КАЭ
@__inference_conv2d_layer_call_and_return_conditional_losses_3893Y*.в+
$в!
К
inputs1(
к "$в!
К
0 
Ъ u
%__inference_conv2d_layer_call_fn_3899L*.в+
$в!
К
inputs1(
к "К Р
?__inference_dense_layer_call_and_return_conditional_losses_5050M╘╒'в$
в
К
inputs	А
к "в
К
0
Ъ h
$__inference_dense_layer_call_fn_5057@╘╒'в$
в
К
inputs	А
к "Кр
L__inference_depthwise_conv2d_1_layer_call_and_return_conditional_losses_1026П`IвF
?в<
:К7
inputs+                            
к "?в<
5К2
0+                            
Ъ ╕
1__inference_depthwise_conv2d_1_layer_call_fn_1042В`IвF
?в<
:К7
inputs+                            
к "2К/+                            с
L__inference_depthwise_conv2d_2_layer_call_and_return_conditional_losses_1236РДIвF
?в<
:К7
inputs+                           @
к "?в<
5К2
0+                           @
Ъ ╣
1__inference_depthwise_conv2d_2_layer_call_fn_1252ГДIвF
?в<
:К7
inputs+                           @
к "2К/+                           @у
L__inference_depthwise_conv2d_3_layer_call_and_return_conditional_losses_1446ТиJвG
@в=
;К8
inputs,                           А
к "@в=
6К3
0,                           А
Ъ ╗
1__inference_depthwise_conv2d_3_layer_call_fn_1462ЕиJвG
@в=
;К8
inputs,                           А
к "3К0,                           А▌
I__inference_depthwise_conv2d_layer_call_and_return_conditional_losses_816П<IвF
?в<
:К7
inputs+                            
к "?в<
5К2
0+                            
Ъ ╡
.__inference_depthwise_conv2d_layer_call_fn_832В<IвF
?в<
:К7
inputs+                            
к "2К/+                            С
A__inference_dropout_layer_call_and_return_conditional_losses_5025L+в(
!в
К
inputs	А
p
к "в
К
0	А
Ъ С
A__inference_dropout_layer_call_and_return_conditional_losses_5030L+в(
!в
К
inputs	А
p 
к "в
К
0	А
Ъ i
&__inference_dropout_layer_call_fn_5035?+в(
!в
К
inputs	А
p
к "К	Аi
&__inference_dropout_layer_call_fn_5040?+в(
!в
К
inputs	А
p 
к "К	Аю
F__inference_functional_1_layer_call_and_return_conditional_losses_3105гE*0123<BCDENTUVW`fghirxyz{ДКЛМНЦЬЭЮЯиоп░▒║└┴┬├╘╒<в9
2в/
%К"
input_1         1(
p

 
к "в
К
0
Ъ ю
F__inference_functional_1_layer_call_and_return_conditional_losses_3290гE*0123<BCDENTUVW`fghirxyz{ДКЛМНЦЬЭЮЯиоп░▒║└┴┬├╘╒<в9
2в/
%К"
input_1         1(
p 

 
к "в
К
0
Ъ э
F__inference_functional_1_layer_call_and_return_conditional_losses_3586вE*0123<BCDENTUVW`fghirxyz{ДКЛМНЦЬЭЮЯиоп░▒║└┴┬├╘╒;в8
1в.
$К!
inputs         1(
p

 
к "в
К
0
Ъ э
F__inference_functional_1_layer_call_and_return_conditional_losses_3771вE*0123<BCDENTUVW`fghirxyz{ДКЛМНЦЬЭЮЯиоп░▒║└┴┬├╘╒;в8
1в.
$К!
inputs         1(
p 

 
к "в
К
0
Ъ ╞
+__inference_functional_1_layer_call_fn_3342ЦE*0123<BCDENTUVW`fghirxyz{ДКЛМНЦЬЭЮЯиоп░▒║└┴┬├╘╒<в9
2в/
%К"
input_1         1(
p

 
к "К╞
+__inference_functional_1_layer_call_fn_3394ЦE*0123<BCDENTUVW`fghirxyz{ДКЛМНЦЬЭЮЯиоп░▒║└┴┬├╘╒<в9
2в/
%К"
input_1         1(
p 

 
к "К┼
+__inference_functional_1_layer_call_fn_3823ХE*0123<BCDENTUVW`fghirxyz{ДКЛМНЦЬЭЮЯиоп░▒║└┴┬├╘╒;в8
1в.
$К!
inputs         1(
p

 
к "К┼
+__inference_functional_1_layer_call_fn_3875ХE*0123<BCDENTUVW`fghirxyz{ДКЛМНЦЬЭЮЯиоп░▒║└┴┬├╘╒;в8
1в.
$К!
inputs         1(
p 

 
к "К█
R__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_1653ДRвO
HвE
CК@
inputs4                                    
к ".в+
$К!
0                  
Ъ ▓
7__inference_global_average_pooling2d_layer_call_fn_1665wRвO
HвE
CК@
inputs4                                    
к "!К                  Ы
A__inference_re_lu_1_layer_call_and_return_conditional_losses_4130V.в+
$в!
К
inputs 
к "$в!
К
0 
Ъ s
&__inference_re_lu_1_layer_call_fn_4135I.в+
$в!
К
inputs 
к "К Ы
A__inference_re_lu_2_layer_call_and_return_conditional_losses_4261V.в+
$в!
К
inputs 
к "$в!
К
0 
Ъ s
&__inference_re_lu_2_layer_call_fn_4266I.в+
$в!
К
inputs 
к "К Ы
A__inference_re_lu_3_layer_call_and_return_conditional_losses_4379V.в+
$в!
К
inputs 
к "$в!
К
0 
Ъ s
&__inference_re_lu_3_layer_call_fn_4384I.в+
$в!
К
inputs 
к "К Ы
A__inference_re_lu_4_layer_call_and_return_conditional_losses_4510V.в+
$в!
К
inputs@
к "$в!
К
0@
Ъ s
&__inference_re_lu_4_layer_call_fn_4515I.в+
$в!
К
inputs@
к "К@Ы
A__inference_re_lu_5_layer_call_and_return_conditional_losses_4628V.в+
$в!
К
inputs@
к "$в!
К
0@
Ъ s
&__inference_re_lu_5_layer_call_fn_4633I.в+
$в!
К
inputs@
к "К@Э
A__inference_re_lu_6_layer_call_and_return_conditional_losses_4759X/в,
%в"
 К
inputsА
к "%в"
К
0А
Ъ u
&__inference_re_lu_6_layer_call_fn_4764K/в,
%в"
 К
inputsА
к "КАЭ
A__inference_re_lu_7_layer_call_and_return_conditional_losses_4877X/в,
%в"
 К
inputsА
к "%в"
К
0А
Ъ u
&__inference_re_lu_7_layer_call_fn_4882K/в,
%в"
 К
inputsА
к "КАЭ
A__inference_re_lu_8_layer_call_and_return_conditional_losses_5008X/в,
%в"
 К
inputsА
к "%в"
К
0А
Ъ u
&__inference_re_lu_8_layer_call_fn_5013K/в,
%в"
 К
inputsА
к "КАЩ
?__inference_re_lu_layer_call_and_return_conditional_losses_4012V.в+
$в!
К
inputs 
к "$в!
К
0 
Ъ q
$__inference_re_lu_layer_call_fn_4017I.в+
$в!
К
inputs 
к "К ╠
"__inference_signature_wrapper_2913еE*0123<BCDENTUVW`fghirxyz{ДКЛМНЦЬЭЮЯиоп░▒║└┴┬├╘╒6в3
в 
,к)
'
input_1К
input_11("$к!

denseК
denseж
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_3881R*в'
 в
К
inputs1(
к "$в!
К
01(
Ъ ~
5__inference_tf_op_layer_ExpandDims_layer_call_fn_3886E*в'
 в
К
inputs1(
к "К1(