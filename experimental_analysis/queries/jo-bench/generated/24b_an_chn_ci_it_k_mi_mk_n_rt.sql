SELECT * FROM name AS n, info_type AS it, aka_name AS an, movie_info AS mi, cast_info AS ci, movie_keyword AS mk, keyword AS k, char_name AS chn, role_type AS rt WHERE ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND it.info = 'release dates' AND k.keyword IN ('hero', 'martial-arts', 'hand-to-hand-combat', 'computer-animated-movie') AND mi.info IS NOT NULL AND (mi.info LIKE 'Japan:%201%' OR mi.info LIKE 'USA:%201%') AND n.gender = 'f' AND n.name LIKE '%An%' AND rt.role = 'actress' AND mi.movie_id = ci.movie_id AND ci.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id AND mk.movie_id = mi.movie_id AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id AND it.id = mi.info_type_id AND mi.info_type_id = it.id AND n.id = ci.person_id AND ci.person_id = n.id AND rt.id = ci.role_id AND ci.role_id = rt.id AND n.id = an.person_id AND an.person_id = n.id AND ci.person_id = an.person_id AND an.person_id = ci.person_id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id AND k.id = mk.keyword_id AND mk.keyword_id = k.id;