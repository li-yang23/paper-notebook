# KG基础知识-OWL语言

owl·web ontology language·不知道为什么不是wol·不是编程语言·网络本体语言，是用来描述things，things的类别和things的关系的语言。owl是一种可计算的基于逻辑的语言，因此用owl表达的知识可以通过计算来进行验证。

owl是w3c语义网络技术栈的一部分，跟RDF，SPARQL一样。

OWL2是用于描述本体的一种语言，本体（ontology）在这里特指某些特定的可计算的东西，通常表示为一个文档。OWL2的基础定义包括：

+ axioms公理：OWL本体论表达的基础陈述

+ entities实体：用于描述现实世界对象的元素

+ expressions：用于根据简单描述组合复杂描述的实体集合

## OWL基本概念

### 类和个体

类用来描述单个实体的特点

> :Mary rdf:type :Person

通过rdf:type 表示个体属于某个类，上面这个栗子里面，说明Mary是Person类的。

类也可以属于某个累，子集关系可以描述为

> :Woman rdfs:subClassOf :Person

子集关系有继承性，即一个类的子集的子集也是它的子集，因此要注意可能有的逻辑错误，比如要是把mother类判定为woman类的子集的话，mother类也会成为person类的子集，而这是不对的，因为没考虑动物

类别也有相同类关系，用equivalentClass说明

> :Person owl:equivalentClass :Human

说明Person和Human是一个类型，说明两者的子集共享

类别的不相交关系表示两个类别没有任何相同的实体，没有相交的子集，描述为

> rdf:type  owl:AllDisjointClasses ;
> 
> owl:members (:Woman :Man) .

类和个体感觉就是数学的类和元素的关系，包括$\in,\notin,\sub$，还有就是不相交$A\cap B=\varnothing$

### 对象的特性

描述两个实体之间的关系，比如

> :John :hasWie :Mary

表示john有个叫Mary的wife。这种写法有顺序性，实体顺序不可调换

两个实体间不存在某个关系表示为 (感觉是一种表示非$\neg$的方法)

> rdf:type  owl:NegativePropertyAssertion ;
> 
> owl:sourceIndividual :Bill ;
> 
> owl:assertionProperty :hasWife ;
> 
> owl :targeIndividual :Mary .

关系也有层次关系，可以表示是另一个关系的子关系

> :hasWife rdfs:subPropertyOf :hasSpouse .

*其实还没太弄明白为什么有些开头是owl，有些是rdf，有些是rdfs，还有些没有*

对关系可以施加范围限制，关系的发出源方施加域限制，关系的目标方施加范围限制，比如

> :hasWife rdfs:domain :Man ;
> 
>                 rdfs:range :Woman ;

域和范围限制都必须是类别，即要求这个关系只能是从一个类指向另一个类的，上面这个就是说只能是woman作为man的wife

个体的相同和不相同关系表示为

> :John owl:differentFrom :Bill .
> 
> :John owl:samAs :Jim .

表示john和bill不是一个人，但和Jim是一个人



个体还有数据类型特性，用来将个体关联到数据值上，这里可以使用多种xml schema datatype。

> :John :hasAge 51 .

看格式主要看后面是不是数值。个体没有某个数据类型特性，可以也可以用类似类和关系方法表示

> rdf:type  owl:NegativePropertyAssertion ;
> 
> owl:sourceIndividual    :Jack ;
> 
> owl:assertionProperty :hasAge ;
> 
> owl:targetValue             53.

数据类型特性也可以进行范围限制，源是个体，可以限制类，目标是数字，可以限制数据类型。即要求这个数据类型特性是某个类别指向某些数据类型的

> :hasAge  rdfs:domain  :Person ;
> 
>                 rdfs:range      xsd:nonNegativeInteger .

上面要求年龄特性只能是人指向非负整数的。

所以说这个类型的子类应该也可以使用这个特性。

## 高级类别关系

### 类别的与，或，非关系（交集，并集，补集）

与关系是同时属于两个类别的类别，比如下面这个

> :Mother owl:equivalentClass [
> 
>     rdf:type  owl:Class ;
> 
>     owl:intersectionOf ( :Woman :Parent)
> 
> ] .

意思就是Mother是Woman和Parent的交集类别

> :Parent owl:equivalentClass [
> 
>     rdf:type  owl:Class ;
> 
>     owl:unionOf ( :Mother :Father )
> 
> ] .

并集关系，意思是Mother和Father都是Parent类的，Parent类是Mother和Father的并集类

> ChildlessPerson owl:equivalentClass [
> 
>     rdf:type  owl:Class ;
> 
>     owl: intersectionOf ( :Person 
> 
>                                           [ rdf:type  owl:Class ;
> 
>                                           owl:complementOf    :Parent])
> 
> ]  .

补集关系，意思是childlessPerson是Parent在Person大类别下的补集类别，对应逻辑中的非关系，即childlessPerson中的元素都不会是Parent

使用子集陈述来表示类别的必要不充分类别

> Grandfather rdfs:subClassof [
> 
>     rdf:type        owl:Class ;
> 
>     owl:intersectionOf (:Man  :Parent )
> 
> ] .

表示grandFather是man和Parent的交集的子集，是grandFather则肯定即是man又是Parent，但反过来则不一定

高级类别也可以直接指定给个体，表明个体和类别的依赖关系，并且过程中不需要给高级类别命名

> :John tdf:type [
> 
>     rdf:type        owl:Class ;
> 
>     owl:intersectionOf ( :Person
> 
>                                         [ rdf:type        owl:Class ;
> 
>                                           owl:complementOf    :Parent])
> 
> ] .

表示John是person，但不是parent

### 特性限制

存在量词（existential quantification）将一个类别定义为与一个特定类别的个体通过特定特性相关联的个体的集合

> :Parent owl:equivalentClass [
> 
>     rdf:type        owl:Restriction ;
> 
>     owl:onProperty    :hasChild ;
> 
>     owl:someValuesFrom       :Person
> 
> ] .

Parent类别的个体通过hasChild特性与Person类别的个体相连。*所以说这两个类别之间是可以包含的，只要特性存在就可以* ，parent类别的个体，至少存在一个有hasChild关联的个体是属于Person类别的

全称量词（universal quantification）将一个类别定义为其个体的所有相关联个体都必须属于一个指定类别

> :HappyPerson    rdf:type    owl:Class ;
> 
>     owl:equivalentClass [
> 
>         rdf:type        owl:Restriction ;
> 
>         owl:onProperty    :hasChild ;
> 
>         owl:allValuesFrom    :Happy
> 
> ] .

一个个体是HappyPerson，当且仅当所有和其有hasChild的个体都属于Happy类别，和存在量词的区别是最后的valuesfrom是allValuesFrom

**特性基数限制**可以在全称和存在量词之外，精细化定义要求的数量

> :John        rdf:type [
> 
>     rdf:type        owl:Restriction ;
> 
>     owl:maxQualifiedCardinality    "4"^^xsd:nonNegativeInteger ;
> 
>     owl:onProperty        :hasChild ;
> 
>     owl:onClass               :Parent
> 
> ] .

    表示John是最多有四个hasChild关系的个体属于Parent，最少就用minQualifiedCardinality，精确数量就用qualifiedCardinality。如果不想讨论有指定关系的个体的类别，可以把onClass去掉

**枚举个体**可以直观定义类别

> :MyBeithdayGuests        owl:equivalentClass [
> 
>     rdf:type        owl:Class ;
> 
>     owl:oneOf    ( :Bill :John :Mary )
> 
> ] .

这种方式精确定义了类别范围，就只有这三个人。因此称之为封闭类别（closed classes）或者枚举集合（enumerated sets）

## 特性高级用法

### 特性的特性

> :hasParent    owl:inverseOf     :hasChild .

    表示hasParent特性和hasChild特性是相反的。在使用时不用真的用hasParent，可以简单声明为hasChild的反特性

> :Orphan    owl:equivalentClass [
> 
>     rdf:type        owl:Restriction ;
> 
>     owl:onProperty    [ owl:inverseOf    :hasChild ] ;
> 
>     owl:allValuesFrom    :Dead
> 
> ] .    

如果是Parent的个体都在Dead类别，那就是Orphan类别

如果特性的两个个体互换位置，特性依然成立，那这个特性就是对称的

> :hasSpouse    rdf:type    owl:SymmetricProperty    .

反之就是不对称的

> :hasChild    rdf:type    owl:AsymmetricProperty    .

如果一个特性可以让个体自己关联自己，那将其称之为反身动词（reflexive反身的）

> :hasRelative    rdf:type    owl:ReflexiveProperty

反之就是非反身的（irreflexive）

> :parentOf    rdf:type    owl:IrreflexiveProperty

两个特性如果不能同时出现在同一对个体上，则称为非相交的

> :hasParent    owl:propertyDisjointWith    :hasSpouse    

一个特性如果限制在一个个体上只能出现一次，则称之为实用的（functional）

> :hasHunband    rdf:type    owl:FunctionalProperty    .

也可以将反特性称之为实用的

> :hasHusband    rdf:typ    owl:InverseFunctionalProperty    .

特性也可以有继承性，对于一个可继承特性，A和B之间存在，B和C之间存在，则A和C之间也存在

> :hasAncestor    rdf:type    owl:TransitiveProperty    .

## 数据类型高级用法

### 使用已有类型定义新数据类型

> :personAge    owl:equivalentClass [
> 
>     rdf:type        rdfs:Datatype ;
> 
>     owl:onDatatype    xsd:integer ;
> 
>     owl:withRestrictions (
> 
>         [ xsd:minInclusive "0"^^xsd:integer ]
> 
>         [xsd:maxInclusive "150"^^xsd:integer ]
> 
>     )
> 
> ] .

将personAge定义为0到150之间的整数

数据类型也可以使用补集，枚举方式定义

> :personAge        owl:equivalentClass [
> 
>     rdf:type        rdfs:Datatype  ;
> 
>     owl:intersectionOf (
> 
>         :personAge
> 
>         [ rdf:type    rdfs:Datatype ;
> 
>           owl:datatypeComplementOf    :minorAge ]
> 
>     )
> 
> ] .

枚举

> :toddlerAge        owl:equivalentClass
> 
> [
> 
>     rdf:type        rdfs:Datatype ;
> 
>     owl:oneOf ( "1"^^xsd:integer   "2"^^xsd:integer )
> 
> ] .

类别可以用数据类型的限制进行定义

> :Teenager    rdfs:subClassOf [
> 
>     rdf:type        owl:Restriction ;
> 
>     owl:onProperty    :hasAge ;
> 
>     owl:sameValuesFrom [
> 
>         rdf:type        rdfs:Datatype ;
> 
>         owl:onDatatype    xsd:integer ;
> 
>         owl:withRestrictions ( [ xsd:minExclusive    "12"^^xsd:integer]
> 
>                                                [ xsd:maxExclusive   "19"^^xsd:integer])
> 
>     ]
> 
> ]

minExclusive 表示开区间的左端点，不包括这个值，Inclusive就是包括这个值的
