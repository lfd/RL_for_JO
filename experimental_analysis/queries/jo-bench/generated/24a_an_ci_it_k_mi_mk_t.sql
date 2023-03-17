SELECT * FROM info_type AS it, keyword AS k, movie_keyword AS mk, title AS t, movie_info AS mi, cast_info AS ci, aka_name AS an WHERE ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND it.info = 'release dates' AND k.keyword IN ('hero', 'martial-arts', 'hand-to-hand-combat') AND mi.info IS NOT NULL AND (mi.info LIKE 'Japan:%201%' OR mi.info LIKE 'USA:%201%') AND t.production_year > 2010 AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = ci.movie_id AND ci.movie_id = t.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND mi.movie_id = ci.movie_id AND ci.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id AND mk.movie_id = mi.movie_id AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id AND it.id = mi.info_type_id AND mi.info_type_id = it.id AND ci.person_id = an.person_id AND an.person_id = ci.person_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id;