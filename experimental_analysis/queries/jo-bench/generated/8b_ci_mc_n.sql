SELECT * FROM movie_companies AS mc, cast_info AS ci, name AS n WHERE ci.note = '(voice: English version)' AND mc.note LIKE '%(Japan)%' AND mc.note NOT LIKE '%(USA)%' AND (mc.note LIKE '%(2006)%' OR mc.note LIKE '%(2007)%') AND n.name LIKE '%Yo%' AND n.name NOT LIKE '%Yu%' AND n.id = ci.person_id AND ci.person_id = n.id AND ci.movie_id = mc.movie_id AND mc.movie_id = ci.movie_id;