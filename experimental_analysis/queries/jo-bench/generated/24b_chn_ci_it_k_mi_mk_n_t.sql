SELECT * FROM name AS n, info_type AS it, movie_info AS mi, movie_keyword AS mk, cast_info AS ci, title AS t, char_name AS chn, keyword AS k WHERE ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND it.info = 'release dates' AND k.keyword IN ('hero', 'martial-arts', 'hand-to-hand-combat', 'computer-animated-movie') AND mi.info IS NOT NULL AND (mi.info LIKE 'Japan:%201%' OR mi.info LIKE 'USA:%201%') AND n.gender = 'f' AND n.name LIKE '%An%' AND t.production_year > 2010 AND t.title LIKE 'Kung Fu Panda%' AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = ci.movie_id AND ci.movie_id = t.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND mi.movie_id = ci.movie_id AND ci.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id AND mk.movie_id = mi.movie_id AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id AND it.id = mi.info_type_id AND mi.info_type_id = it.id AND n.id = ci.person_id AND ci.person_id = n.id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id AND k.id = mk.keyword_id AND mk.keyword_id = k.id;