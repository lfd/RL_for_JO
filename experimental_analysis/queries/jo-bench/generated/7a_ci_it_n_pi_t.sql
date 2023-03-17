SELECT * FROM title AS t, cast_info AS ci, info_type AS it, person_info AS pi, name AS n WHERE it.info = 'mini biography' AND n.name_pcode_cf BETWEEN 'A' AND 'F' AND (n.gender = 'm') AND pi.note = 'Volker Boehm' AND t.production_year BETWEEN 1980 AND 1995 AND n.id = pi.person_id AND pi.person_id = n.id AND ci.person_id = n.id AND n.id = ci.person_id AND t.id = ci.movie_id AND ci.movie_id = t.id AND it.id = pi.info_type_id AND pi.info_type_id = it.id AND pi.person_id = ci.person_id AND ci.person_id = pi.person_id;