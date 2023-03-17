SELECT * FROM name AS n, cast_info AS ci, movie_info AS mi, info_type AS it1, movie_companies AS mc, title AS t WHERE ci.note IN ('(writer)', '(head writer)', '(written by)', '(story)', '(story editor)') AND it1.info = 'genres' AND mc.note LIKE '%(Blu-ray)%' AND mi.info IN ('Horror', 'Thriller') AND n.gender = 'm' AND t.production_year > 2000 AND (t.title LIKE '%Freddy%' OR t.title LIKE '%Jason%' OR t.title LIKE 'Saw%') AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = ci.movie_id AND ci.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND ci.movie_id = mi.movie_id AND mi.movie_id = ci.movie_id AND ci.movie_id = mc.movie_id AND mc.movie_id = ci.movie_id AND mi.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id AND n.id = ci.person_id AND ci.person_id = n.id AND it1.id = mi.info_type_id AND mi.info_type_id = it1.id;