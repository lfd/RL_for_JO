SELECT * FROM movie_companies AS mc, aka_title AS at, title AS t, movie_info AS mi, movie_keyword AS mk WHERE mc.note LIKE '%(200%)%' AND mc.note LIKE '%(worldwide)%' AND mi.note LIKE '%internet%' AND mi.info LIKE 'USA:% 200%' AND t.production_year BETWEEN 2005 AND 2010 AND t.id = at.movie_id AND at.movie_id = t.id AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mk.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND mk.movie_id = at.movie_id AND at.movie_id = mk.movie_id AND mi.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id AND mi.movie_id = at.movie_id AND at.movie_id = mi.movie_id AND mc.movie_id = at.movie_id AND at.movie_id = mc.movie_id;