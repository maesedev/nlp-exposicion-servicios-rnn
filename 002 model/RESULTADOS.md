**Observaciones de resultados:**

Temperatura baja (0.5): Nombres cortos y más "limpios" — Trisaurus, Velapelta, Megala.
Temperatura alta (2.0): Nombres más largos, raros y creativos — Velwiarondemryxtrihnko, Braphnehigosauruskoceandus, Rexorgenheranchhhocaus.
Anomalía con sau: El modelo ignoró completamente el prefijo y generó nombres que empiezan con U (Utitan, Uinodon, Usaurus). Esto sugiere un posible bug en el tokenizador o en cómo el modelo fue entrenado con ese prefijo específico — "sau" probablemente se tokeniza como una unidad que el modelo mapea internamente a otra representación.

| Prefijo |	Temperatura	 | Nombre generado
1	kal	0.5	Kalavenator
2	kal	1.0	Kalglong
3	kal	1.5	Kalmoefia
4	kal	2.0	Kaldayuansu
5	rex	0.5	Rexcanoceratops
6	rex	1.0	Rexingpaar
7	rex	1.5	Rexorgenheranchhhocaus
8	rex	2.0	Rexaltigtandia
9	vel	0.5	Velapelta
10	vel	1.0	Velemis
11	vel	1.5	Velaetoceratops
12	vel	2.0	Velwiarondemryxtrihnko
13	tri	0.5	Trisaurus
14	tri	1.0	Trigyadrorox
15	tri	1.5	Trieosaurus
16	tri	2.0	Trictsosaurustyraeus
17	sau	0.5	Utitan ⚠️
18	sau	1.0	Uinodon ⚠️
19	sau	1.5	Usaurus ⚠️
20	sau	2.0	Ulchenlmpalchor ⚠️
21	meg	0.5	Megala
22	meg	1.0	Megysaurus
23	meg	1.5	Meginosaurus
24	meg	2.0	Megritodon
25	pte	0.5	Pteeos
26	pte	1.0	Pterenoxafesor
27	pte	1.5	Ptesichunsaurus
28	pte	2.0	Ptenosaurus
29	al	0.5	Almamis
30	al	1.0	Alwinisaurus
31	al	1.5	Alcotit
32	al	2.0	Algarsaurorfanus
33	bra	0.5	Brasaurus
34	bra	1.0	Bramrosaurus
35	bra	1.5	Brasidais
36	bra	2.0	Braphnehigosauruskoceandus
37	top	0.5	Topelosaurus
38	top	1.0	Topraptor
39	top	1.5	Topladrovia
40	top	2.0	Topllasauria


MODEL Phi4 14B (https://ollama.com/library/phi4)

Prompt: 

